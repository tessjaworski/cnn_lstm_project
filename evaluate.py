import torch
import torch.nn as nn
from model import HybridCNNLSTM
from dataloader import load_dataset

# load data
(_, _, _,
 _, _, _,
 X_era5_test, X_cora_test, y_test) = load_dataset()

device = "cpu"

class StormSurgeDataset(torch.utils.data.Dataset):
    def __init__(self, x_era5, x_cora, y):
        self.x_era5 = torch.from_numpy(x_era5).float()
        self.x_cora = torch.from_numpy(x_cora).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.x_era5)

    def __getitem__(self, idx):
        return self.x_era5[idx], self.x_cora[idx], self.y[idx]
    
# load model
model = HybridCNNLSTM(
    era5_channels=X_era5_test.shape[2],
    zeta_nodes=y_test.shape[1]
).to(device)
model.load_state_dict(torch.load("cnn_lstm_model.pth", map_location=device))
model.eval()

# evaluation
test_dataset = StormSurgeDataset(X_era5_test, X_cora_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

criterion = nn.MSELoss()
mse_total, mae_total = 0.0, 0.0

with torch.no_grad():
    for x_era5, x_cora, y_true in test_loader:
        x_era5, x_cora, y_true = (
            x_era5.to(device),
            x_cora.to(device),
            y_true.to(device)
        )
        y_pred = model(x_era5, x_cora)
        mse = criterion(y_pred, y_true)
        mae = torch.mean(torch.abs(y_pred - y_true))
        mse_total += mse.item()
        mae_total += mae.item()

print(f"Test MSE: {mse_total / len(test_loader):.4f}")
print(f"Test MAE: {mae_total / len(test_loader):.4f}")
