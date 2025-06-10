import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from model import HybridCNNLSTM
from dataloader import load_dataset, SEQ_LEN, PRED_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
era5_mm, cora, _, _, test_idx, mask_np = load_dataset()
mask = torch.from_numpy(mask_np).to(device)

# persistence baseline (3-hour forecast)
y0 = cora[test_idx]
yN = cora[test_idx + PRED_LEN]
persistence_mse = np.mean((yN - y0) ** 2)
print(f"{PRED_LEN}-h persistence MSE: {persistence_mse:.4f}")

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

test_ds     = StormSurgeDataset(era5_mm, cora, test_idx)
test_loader = data.DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

model = HybridCNNLSTM(
    era5_channels=era5_mm.shape[1],
    zeta_nodes=int(mask.sum()),
    pred_steps=PRED_LEN
).to(device)
model.load_state_dict(torch.load("best_model_3h_normalized.pth", map_location=device))
model.eval()

criterion = nn.MSELoss()
mse_total = 0.0
mae_total = 0.0
all_pred, all_true = [], []

with torch.no_grad():
    for x5, xz, y_true in test_loader:
        x5, xz, y_true = x5.to(device), xz.to(device), y_true.to(device)
        x5 = torch.nan_to_num(x5); xz = torch.nan_to_num(xz); y_true = torch.nan_to_num(y_true)
        y_pred = model(x5, xz)
        mse_total += criterion(y_pred, y_true).item()
        mae_total += torch.mean(torch.abs(y_pred - y_true)).item()
        all_pred.append(y_pred.cpu().numpy().ravel())
        all_true.append(y_true.cpu().numpy().ravel())

n = len(test_loader)
print(f"Test MSE: {mse_total / n:.4f}")
print(f"Test MAE: {mae_total / n:.4f}")

flat_pred = np.concatenate(all_pred)
flat_true = np.concatenate(all_true)
print("R²:", r2_score(flat_true, flat_pred))

plt.figure(figsize=(6,6))
plt.scatter(flat_true, flat_pred, s=1, alpha=0.3)
lims = [flat_true.min(), flat_true.max()]
plt.plot(lims, lims, 'k--', lw=1)
plt.xlabel("True ζ")
plt.ylabel("Predicted ζ")
plt.title("Predicted vs. True ζ for 6 hr Prediction")
plt.tight_layout()
plt.savefig("scatter_zeta_test.png", dpi=150)
print("Saved 6hr_scatter_zeta_test.png")