import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ======== CONFIG ========
SEQ_LEN = 64
BATCH_SIZE = 64
EPOCHS = 3
TRAIN_PATH = "data/processed_data/cleaned/training_final.csv"
TEST_PATH = "data/processed_data/cleaned/testing_final.csv"
RED_TEAM_PATH = "data/processed_data/cleaned/redteam.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======== DATASET ========
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.data = data
        self.samples = self.create_sequences(data, seq_len)

    def create_sequences(self, data, seq_len):
        return [data[i : i + seq_len] for i in range(len(data) - seq_len)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)


def load_csv_as_tensor(path, seq_len, scaler=None, fit_scaler=True):
    df = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    if scaler is None:
        scaler = MinMaxScaler()

    features = (
        scaler.fit_transform(df.values) if fit_scaler else scaler.transform(df.values)
    )
    return TimeSeriesDataset(features, seq_len), scaler, df


# ======== MODEL ========
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        lat_out, _ = self.latent(enc_out)
        dec_out, _ = self.decoder(lat_out)
        return self.output_layer(dec_out)


# ======== LOAD DATA ========
train_dataset, scaler, _ = load_csv_as_tensor(TRAIN_PATH, SEQ_LEN)
test_dataset, _, test_df = load_csv_as_tensor(
    TEST_PATH, SEQ_LEN, scaler, fit_scaler=False
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======== TRAIN ========
model = LSTMAutoencoder(
    input_dim=train_dataset[0].shape[1], hidden_dim=64, latent_dim=32
).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        batch = batch.to(DEVICE)
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(train_loader):.6f}")

# ======== EVALUATE ========
model.eval()
errors = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        output = model(batch)
        loss = torch.mean((output - batch) ** 2, dim=(1, 2))
        errors.extend(loss.cpu().numpy())

errors = np.array(errors)
threshold = np.percentile(errors, 95)
anomalies = (errors > threshold).astype(int)

# ======== LOAD & TAG RED TEAM EVENTS ========
red_df = pd.read_csv(RED_TEAM_PATH).astype(str)
test_df = test_df.astype(str)
test_df["is_malicious"] = 0

for _, event in red_df.iterrows():
    match = (
        (test_df["time"] == event["time"])
        & (test_df["src_comp"] == event["src_comp"])
        & (test_df["dst_comp"] == event["dst_comp"])
        & (test_df["src_user"] == event["src_user"])
        & (test_df["src_domain"] == event["src_domain"])
    )
    test_df.loc[match, "is_malicious"] = 1

# ======== LABEL SEQUENCES ========
labels = []
mal_flag = test_df["is_malicious"].astype(int).values
for i in range(len(test_df) - SEQ_LEN):
    labels.append(1 if np.any(mal_flag[i : i + SEQ_LEN]) else 0)

y_true = np.array(labels[: len(anomalies)])
y_pred = anomalies

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n--- Evaluation Metrics ---")
print(f"Threshold (95th percentile): {threshold:.6f}")
print(f"Anomalies detected: {np.sum(anomalies)} / {len(anomalies)} sequences")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ======== SAVE MODEL ========
torch.save(model, "lstm_autoencoder.pth")
