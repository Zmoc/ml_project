from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.data = data
        self.samples = self.create_sequences(data, seq_len)

    def create_sequences(self, data, seq_len):
        sequences = []
        for i in range(len(data) - seq_len):
            seq = data[i : i + seq_len]
            sequences.append(seq)
        return sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)


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
        out = self.output_layer(dec_out)
        return out


def load_csv_as_tensor(path, seq_len):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df.values)
    return TimeSeriesDataset(features, seq_len), scaler


def scan_traffic(csv_path, model_path, threshold=5, seq_len=30, batch_size=64):
    print("Loading Datasets")
    test_dataset, _ = load_csv_as_tensor(csv_path, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = LSTMAutoencoder(
        input_dim=test_dataset[0].shape[1], hidden_dim=64, latent_dim=32
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    errors = []
    all_outputs = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = torch.mean((output - batch) ** 2, dim=(1, 2))
            errors.extend(loss.cpu().numpy())
            all_outputs.extend(output.cpu().numpy())

    errors = np.array(errors)
    threshold = np.percentile(errors, 100 - threshold)

    anomalies = (errors > threshold).astype(int)

    print(f"Anomalies detected: {np.sum(anomalies)} out of {len(anomalies)} sequences")
    print(f"Anomaly rate: {np.sum(anomalies)/len(anomalies):.2%}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H_%M_%S")

    anomaly_indices = np.where(anomalies == 1)[0]
    with open(f"log/anomaly-{timestamp}.txt", "w") as f:
        for i in anomaly_indices:
            f.write(f"Anomaly at sequence starting index: {i}\n")
