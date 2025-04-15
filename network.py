import numpy as np
import torch
from torch.utils.data import DataLoader

from etc.network_detect_model import LSTMAutoencoder, load_csv_as_tensor


def scan_traffic(csv_path, seq_len=30, batch_size=64):
    print("Loading Datasets")
    test_dataset, _ = load_csv_as_tensor(csv_path, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = LSTMAutoencoder(
        input_dim=test_dataset[0].shape[1], hidden_dim=64, latent_dim=32
    ).to(device)

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
    threshold = np.percentile(errors, 95)

    anomalies = (errors > threshold).astype(int)
    print(f"Anomalies detected: {np.sum(anomalies)} out of {len(anomalies)} sequences")

    anomaly_indices = np.where(anomalies == 1)[0]
    with open("log/anomaly_log.txt", "w") as f:
        for i in anomaly_indices:
            f.write(f"Anomaly at sequence starting index: {i}\n")
