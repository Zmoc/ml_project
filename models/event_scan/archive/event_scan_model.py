# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.event_scan.archive.time_series_dataset import TimeSeriesDataset
from torch.nn import MSELoss

# %%
# ======== CONFIG ========
SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 3
TRAIN_PATH = (
    "D:/GitRepos/ml_project/data/event/processed_data/cleaned/training_final.csv"
)
TEST_PATH = "D:/GitRepos/ml_project/data/event/processed_data/cleaned/testing_final.csv"
RED_TEAM_PATH = "data/event/processed_data/cleaned/redteam.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%


# %%
# ======== MODEL ========
class CNNLSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dims,
        cnn_channels=64,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.2,
    ):
        super(CNNLSTMAutoencoder, self).__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=cardinality, embedding_dim=dim)
                for cardinality, dim in emb_dims
            ]
        )
        self.emb_total_dim = sum(dim for _, dim in emb_dims)

        self.input_dim = input_dim + self.emb_total_dim

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.encoder_lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.decoder_lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=cnn_channels,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.decoder_cnn = nn.Sequential(
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, self.input_dim, kernel_size=3, padding=1),
        )

    def forward(self, x_cont, x_cat):
        """
        x_cont: [batch, seq_len, num_cont_features]
        x_cat: list of [batch, seq_len] tensors for each categorical feature
        """
        embedded = [emb(cat) for emb, cat in zip(self.embeddings, x_cat)]
        embedded_cat = torch.cat(embedded, dim=2)

        x = torch.cat([x_cont, embedded_cat], dim=2)

        x = x.permute(0, 2, 1)
        x = self.encoder_cnn(x)
        x = x.permute(0, 2, 1)

        enc_out, (h, c) = self.encoder_lstm(x)

        dec_out, _ = self.decoder_lstm(enc_out)

        dec_out = dec_out.permute(0, 2, 1)
        x_recon = self.decoder_cnn(dec_out)
        x_recon = x_recon.permute(0, 2, 1)
        return x_recon


# %%
# ======== TRAIN ========
def reconstruction_loss(output, x_cont):
    return torch.mean((output[:, :, : x_cont.shape[2]] - x_cont) ** 2, dim=(1, 2))


if __name__ == "__main__":
    # ======== DATASET ========
    import os
    import joblib

    PREPROCESSED_PATH = "preprocessed_train.parquet"
    PREPROCESSOR_PATH = "preprocessor.pkl"

    class FastPreloadedDataset(Dataset):
        def __init__(self, data, seq_len, cont_len):
            self.cont_len = cont_len
            self.num_cont_features = cont_len  # <-- Added this line
            self.samples = self.create_sequences(data, seq_len)

        def create_sequences(self, data, seq_len):
            return [data[i : i + seq_len] for i in range(len(data) - seq_len + 1)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            cont = sample[:, : self.cont_len]
            cat = sample[:, self.cont_len :]
            return {
                "cont": torch.tensor(cont, dtype=torch.float32),
                "cat": [
                    torch.tensor(cat[:, i], dtype=torch.long)
                    for i in range(cat.shape[1])
                ],
            }

    # === COLUMN DEFINITIONS ===
    low_dim_columns = [
        "auth_type",
        "logon_type",
        "auth_orient",
        "pass_fail",
        "event_type",
        "prtcl",
        "start/end",
    ]
    high_dim_columns = [
        "src_comp",
        "dst_comp",
        "src_user",
        "src_domain",
        "dst_user",
        "dst_domain",
        "src_compr",
        "comp_rsvd",
        "src_port",
        "dst_port",
        "proc_name",
    ]
    num_columns = ["byte_cnt", "pckt_cnt", "dur"]

    dtype_map = {col: "category" for col in low_dim_columns + high_dim_columns}
    dtype_map.update({col: "float32" for col in num_columns})

    # === LOAD TRAIN DATA ===
    if os.path.exists(PREPROCESSED_PATH) and os.path.exists(PREPROCESSOR_PATH):
        print("Loading preprocessed data from Parquet...")
        df_out = pd.read_parquet(PREPROCESSED_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        cont_len = len([col for col in df_out.columns if not col.startswith("high_")])
        cat_cols = [col for col in df_out.columns if col.startswith("high_")]

        X_processed = df_out.iloc[:, :cont_len].astype(np.float32).to_numpy()
        encoded_high_dim = df_out[cat_cols].astype(np.int64).to_numpy()
        data = np.concatenate([X_processed, encoded_high_dim], axis=1)

        train_dataset = FastPreloadedDataset(data, SEQ_LEN, cont_len)
        num_cont_features = cont_len

        # For embedding dims, you still need ref_df
        ref_df = pd.read_csv(
            TRAIN_PATH,
            usecols=high_dim_columns,
            dtype={col: "category" for col in high_dim_columns},
        )
        emb_dims = [
            (len(ref_df[col].astype("category").cat.categories), 64)
            for col in high_dim_columns
        ]
    else:
        print("Processing raw CSVs...")
        train_df = pd.read_csv(
            TRAIN_PATH,
            usecols=low_dim_columns + high_dim_columns + num_columns,
            dtype=dtype_map,
            low_memory=False,
        )
        train_df = train_df.iloc[:5000]
        test_df = pd.read_csv(
            TEST_PATH,
            usecols=low_dim_columns + high_dim_columns + num_columns,
            dtype=dtype_map,
            low_memory=False,
        )

        train_dataset = TimeSeriesDataset(
            train_df,
            seq_len=SEQ_LEN,
            is_train=True,
            ref_df=train_df,
            low_dim_columns=low_dim_columns,
            high_dim_columns=high_dim_columns,
            num_columns=num_columns,
        )
        test_dataset = TimeSeriesDataset(
            test_df,
            seq_len=SEQ_LEN,
            is_train=False,
            ref_df=train_df,
            low_dim_columns=low_dim_columns,
            high_dim_columns=high_dim_columns,
            num_columns=num_columns,
            preprocessor=train_dataset.preprocessor,
        )

        # Save for reuse
        print("Saving preprocessed train data for reuse...")
        all_cols = train_dataset.preprocessor.get_feature_names_out().tolist() + [
            f"high_{col}" for col in high_dim_columns
        ]
        df_out = pd.DataFrame(train_dataset.data, columns=all_cols)
        df_out.to_parquet(PREPROCESSED_PATH, index=False)
        joblib.dump(train_dataset.preprocessor, PREPROCESSOR_PATH)

        emb_dims = train_dataset.emb_dims
        num_cont_features = train_dataset.num_cont_features

    # === Ensure test_dataset exists (either way) ===
    if "test_dataset" not in locals():
        test_df = pd.read_csv(
            TEST_PATH,
            usecols=low_dim_columns + high_dim_columns + num_columns,
            dtype=dtype_map,
            low_memory=False,
        )
        train_df = pd.read_csv(
            TRAIN_PATH,
            usecols=high_dim_columns,
            dtype={col: "category" for col in high_dim_columns},
            low_memory=False,
        )
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        test_dataset = TimeSeriesDataset(
            test_df,
            seq_len=SEQ_LEN,
            is_train=False,
            ref_df=train_df,
            low_dim_columns=low_dim_columns,
            high_dim_columns=high_dim_columns,
            num_columns=num_columns,
            preprocessor=preprocessor,
        )

    # emb_dims = train_dataset.emb_dims

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        # num_workers=multiprocessing.cpu_count() // 2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        # num_workers=multiprocessing.cpu_count() // 2,
    )
    print("Data ready to load")

    def train_model(model, train_loader, optimizer, criterion, device, epochs=EPOCHS):
        model.to(device)
        model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # === Unpack input ===
                x_cont = batch["cont"].to(device)  # [B, T, num_cont]
                x_cat = [cat.to(device) for cat in batch["cat"]]  # List of [B, T]

                # === Forward Pass ===
                optimizer.zero_grad()
                x_recon = model(x_cont, x_cat)

                # === Loss (only on continuous input) ===
                loss = reconstruction_loss(output, x_cont)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

    # Calculate the number of continuous features (used in loss)
    num_cont_features = (
        train_dataset.num_cont_features
    )  # Assuming you track this in your dataset

    # Instantiate model
    model = CNNLSTMAutoencoder(
        input_dim=num_cont_features,
        emb_dims=emb_dims,
        cnn_channels=64,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.2,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()

    # Train
    train_model(model, train_loader, optimizer, criterion, DEVICE, epochs=EPOCHS)

    # %%
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in test_loader:
            x_cont = batch["cont"].to(DEVICE)  # [B, T, num_cont]
            x_cat = [cat.to(DEVICE) for cat in batch["cat"]]  # List of [B, T]

            output = model(x_cont, x_cat)

            # Reconstruction loss per sample
            loss = reconstruction_loss(output, x_cont)

            errors.extend(loss.cpu().numpy())

    # Compute anomaly threshold and predictions
    errors = np.array(errors)
    threshold = np.percentile(errors, 95)  # top 5% as anomalies
    anomalies = (errors > threshold).astype(int)

    # %%
    # ======== SAVE MODEL ========
    torch.save(model.state_dict(), "lstm_autoencoder_weights.pth")
