import torch
import torch.nn as nn
import torch.utils.data as data
from model import HybridCNNLSTM
from dataloader import load_dataset, SEQ_LEN, PRED_LEN
from cora_graph import load_cora_coordinates

# load data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
era5_mm, cora, _, _, test_idx, mask_np = load_dataset()
mask  = torch.from_numpy(mask_np).to(device)
coords = load_cora_coordinates("/home/exouser/Jan2015_cropped.nc", mask_np)

class StormSurgeDataset(torch.utils.data.Dataset):
    def __init__(self, era5_mm, cora_arr, start_idx):
        self.era5 = era5_mm        
        self.cora = cora_arr        
        self.idxs = start_idx 

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx]
        x_era5 = self.era5[i : i + SEQ_LEN]     # (T,C,H,W)
        x_cora = self.cora[i : i + SEQ_LEN]     # (T,WetNodes)
        y      = self.cora[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]         # (WetNodes,)
        return (torch.tensor(x_era5, dtype=torch.float32),
                torch.tensor(x_cora, dtype=torch.float32),
                torch.tensor(y,      dtype=torch.float32))

test_ds  = StormSurgeDataset(era5_mm, cora, test_idx)
test_loader = data.DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

# load model
model = HybridCNNLSTM(
    era5_channels=era5_mm.shape[1],
    zeta_nodes=mask.sum().item(),
    coords = coords,
    k_neighbors = 8,
    pred_steps = PRED_LEN
).to(device)

model.load_state_dict(torch.load("/home/exouser/cnn_lstm_project/best_model.pth", map_location=device))
model.eval()

criterion = nn.MSELoss()
mse_total, mae_total = 0.0, 0.0

with torch.no_grad():
    for x_era5, x_cora, y_true in test_loader:
        x_era5, x_cora, y_true = (
            x_era5.to(device),
            x_cora.to(device),
            y_true.to(device)
        )
        x_era5 = torch.nan_to_num(x_era5, nan=0.0)
        x_cora = torch.nan_to_num(x_cora, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)

        y_pred = model(x_era5, x_cora)
        mse = criterion(y_pred, y_true)
        mae  = torch.mean(torch.abs(y_pred - y_true))
        mse_total += mse.item()
        mae_total += mae.item()

print(f"Test MSE: {mse_total / len(test_loader):.4f}")
print(f"Test MAE: {mae_total / len(test_loader):.4f}")
