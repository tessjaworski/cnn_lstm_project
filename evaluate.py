import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from model import CNN_GNN_Hybrid
from dataloader import load_dataset, CORA_PATHS, SEQ_LEN, PRED_LEN
from cora_graph      import load_cora_coordinates, build_edge_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
era5_mm, cora_norm, tr_idx, va_idx, test_idx, mask_np, μ_cora, σ_cora = load_dataset()
μ_cora = torch.from_numpy(μ_cora).float().to(device)
σ_cora = torch.from_numpy(σ_cora).float().to(device)
mask = torch.from_numpy(mask_np).to(device)

coords     = load_cora_coordinates(CORA_PATHS[0], mask_np)
edge_index = build_edge_index(coords, k=8).to(device)

# persistence baseline (3-hour forecast)
y0 = cora_norm[test_idx]
yN = cora_norm[test_idx + PRED_LEN]
persist_mse = np.mean((yN - y0) ** 2)
print(f"{PRED_LEN}-h persistence MSE (norm): {persist_mse:.4f}")

class StormSurgeDataset(data.Dataset):
    def __init__(self, era5, cora, idxs):
        self.era5, self.cora, self.idxs = era5, cora, idxs
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        start = self.idxs[i]
        x5 = self.era5[start:start+SEQ_LEN]
        xz = self.cora[start:start+SEQ_LEN]
        y  = self.cora[start+SEQ_LEN:start+SEQ_LEN+PRED_LEN]
        return (
            torch.tensor(x5, dtype=torch.float32),
            torch.tensor(xz, dtype=torch.float32),
            torch.tensor(y,  dtype=torch.float32),
        )

test_ds     = StormSurgeDataset(era5_mm, cora_norm, test_idx)
test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

num_era5_feats = era5_mm.shape[1]
model = CNN_GNN_Hybrid(
    era5_channels    = era5_mm.shape[1],
    cnn_hidden       = 32,
    cnn_lstm_hidden  = 128,
    gcn_hidden       = 32,
    zeta_lstm_hidden = 64,
    pred_steps       = PRED_LEN
).to(device)
model.load_state_dict(torch.load("gnn_model_24h_normalized.pth", map_location=device))
model.eval()

criterion = nn.MSELoss()
mse_total = 0.0
mae_total = 0.0
all_pred, all_true = [], []

with torch.no_grad():
    for x5, xz, y_true in test_loader:
        x5, xz, y_true = x5.to(device), xz.to(device), y_true.to(device)
        x5 = torch.nan_to_num(x5); xz = torch.nan_to_num(xz); y_true = torch.nan_to_num(y_true)

        B, T, C, H, W = x5.shape
        era5_seq = x5

        y_pred = model(era5_seq, xz, edge_index)

        y_pred = y_pred * σ_cora + μ_cora
        y_true = y_true * σ_cora + μ_cora

        mse_total += criterion(y_pred, y_true).item()
        mae_total += torch.mean(torch.abs(y_pred - y_true)).item()
        all_pred.append(y_pred.cpu().numpy())
        all_true.append(y_true.cpu().numpy())
    all_pred = np.concatenate(all_pred, axis=0)  # shape: [total_samples, T, N]
    all_true = np.concatenate(all_true, axis=0)

n = len(test_loader)
print(f"Test MSE: {mse_total / n:.4f}")
print(f"Test MAE: {mae_total / n:.4f}")

flat_pred = np.concatenate(all_pred)
flat_true = np.concatenate(all_true)
print("R² (norm):", r2_score(flat_true, flat_pred))

plt.figure(figsize=(6,6))
plt.scatter(flat_true, flat_pred, s=1, alpha=0.3)
lims = [flat_true.min(), flat_true.max()]
plt.plot(lims, lims, 'k--', lw=1)
plt.xlabel("True ζ")
plt.ylabel("Predicted ζ")
plt.title("Predicted vs. True ζ for 24 hr Prediction")
plt.tight_layout()
plt.savefig("24hr_gnn_normalized_scatter_zeta_test.png", dpi=150)
print("Saved 24hr_gnn_normalized_scatter_zeta_test.png")

selected_frames = [5, 11, 23]  # show first 3 timesteps (adjust as needed)
coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
num_nodes = int(mask_np.sum())
pred_array = np.concatenate(all_pred).reshape(-1, num_nodes)
true_array = np.concatenate(all_true).reshape(-1, num_nodes)

nodes_to_plot = [40, 90, 160]
sample_idx = 0

for node_idx in nodes_to_plot:
    lat, lon = coords_np[node_idx]
    pred_24 = all_pred[sample_idx, :, node_idx]
    true_24 = all_true[sample_idx, :, node_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(24), true_24, label='Ground Truth ζ')
    plt.plot(np.arange(24), pred_24, label='Predicted ζ')
    plt.xlabel("Forecast Hour")
    plt.ylabel("ζ (meters)")
    plt.title(f"Node {node_idx} ({lat:.3f}N, {lon:.3f}E) – 24h Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"time_series_node{node_idx}.png", dpi=150)
    plt.close()
    print(f"Saved time_series_node{node_idx}.png")
