import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt


# ---- CONFIG ----
SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 3
TRAIN_CSV = "data/event/processed_data/cleaned/training_final.csv"
TEST_CSV = "data/event/processed_data/cleaned/testing_final.csv"
RED_CSV = "data/event/processed_data/cleaned/redteam.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROWS = 20000

ENCODED_DIR = "cached_encoded"
os.makedirs(ENCODED_DIR, exist_ok=True)


# 1) LOAD & PREPROCESS (or use cached)
data_train_path = f"{ENCODED_DIR}/data_train.npy"
data_test_path = f"{ENCODED_DIR}/data_test.npy"
y_event_path = f"{ENCODED_DIR}/y_event.npy"
ohe_path = f"{ENCODED_DIR}/onehot_encoder.pkl"
scaler_path = f"{ENCODED_DIR}/scaler.pkl"

id_cols = ["time", "src_comp", "dst_comp", "src_user", "src_domain"]
auth_cols = [
    "auth_type",
    "logon_type",
    "auth_orient",
    "pass_fail",
    "event_type",
    "prtcl",
    "start/end",
]
flow_cols = ["byte_cnt", "pckt_cnt", "dur"]

if os.path.exists(data_train_path) and os.path.exists(data_test_path):
    print("Loading cached encoded data...")
    data_train = np.load(data_train_path)
    data_test = np.load(data_test_path)
    y_event = np.load(y_event_path)
    ohe = joblib.load(ohe_path)
    scaler = joblib.load(scaler_path)
else:
    print("Encoding data from scratch...")
    df_train = pd.read_csv(TRAIN_CSV).head(ROWS)
    df_test_raw = pd.read_csv(TEST_CSV, usecols=id_cols + auth_cols + flow_cols).head(
        ROWS
    )
    df_red = pd.read_csv(RED_CSV, dtype=str)

    df_test_raw["is_malicious"] = 0
    for _, r in df_red.iterrows():
        mask = (
            (df_test_raw["time"].astype(str) == r["time"])
            & (df_test_raw["src_comp"] == r["src_comp"])
            & (df_test_raw["dst_comp"] == r["dst_comp"])
            & (df_test_raw["src_user"] == r["src_user"])
            & (df_test_raw["src_domain"] == r["src_domain"])
        )
        df_test_raw.loc[mask, "is_malicious"] = 1

    df_test = df_test_raw[auth_cols + flow_cols]
    y_event = df_test_raw["is_malicious"].values

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    A_train = ohe.fit_transform(df_train[auth_cols])
    scaler = StandardScaler()
    F_train = scaler.fit_transform(df_train[flow_cols])
    data_train = np.hstack([A_train, F_train]).astype(np.float32)

    A_test = ohe.transform(df_test[auth_cols])
    F_test = scaler.transform(df_test[flow_cols])
    data_test = np.hstack([A_test, F_test]).astype(np.float32)

    np.save(data_train_path, data_train)
    np.save(data_test_path, data_test)
    np.save(y_event_path, y_event)
    joblib.dump(ohe, ohe_path)
    joblib.dump(scaler, scaler_path)


# 2) BUILD SEQUENCE LABELS for test
seq_labels = [
    1 if y_event[i : i + SEQ_LEN].any() else 0
    for i in range(len(y_event) - SEQ_LEN + 1)
]


# 3) SEQ DATASET & DATALOADERS
class SeqDataset(Dataset):
    def __init__(self, data, seq_len):
        self.samples = [data[i : i + seq_len] for i in range(len(data) - seq_len + 1)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)


train_ds = SeqDataset(data_train, SEQ_LEN)
test_ds = SeqDataset(data_test, SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

input_dim = data_train.shape[1]


# 4) MODEL DEFINITION
class SimpleLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.decoder = nn.LSTM(
            hidden_dim, input_dim, num_layers=num_layers, batch_first=True
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        dec_in = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        recon, _ = self.decoder(dec_in)
        return recon


model = SimpleLSTMAutoencoder(input_dim, num_layers=2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 5) TRAIN
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} avg loss: {total_loss/len(train_loader):.6f}")


# 6) EVALUATE ON TEST & COMPUTE METRICS
model.eval()
errors = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        recon = model(batch)
        seq_err = torch.mean((recon - batch) ** 2, dim=(1, 2))
        errors.extend(seq_err.cpu().numpy())

errors = np.array(errors)
threshold = np.percentile(errors, 95)
preds = (errors > threshold).astype(int)
y_true = np.array(seq_labels, dtype=int)
false_positives = np.sum((preds == 1) & (y_true == 0))

precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
f1 = f1_score(y_true, preds)

print(f"Threshold: {threshold:.4f}")
print(f"Anomalies: {preds.sum()}/{len(preds)} sequences")
print(f"False Positives: {false_positives}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print(f"Total test sequences: {len(test_ds)}")
print(f"Labeled red team sequences: {sum(y_true)}")

plt.hist(errors, bins=50)
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.title("Reconstruction Error Distribution")
plt.xlabel("MSE")
plt.ylabel("Count")
plt.legend()
plt.show()

# 7) SAVE MODEL
torch.save(model.state_dict(), "simple_lstm_autoencoder.pth")
