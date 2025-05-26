# loads dataset from dataloader.py
# initializes cnn+lstm model from model.py
# define loss function and optimizer
# run the training loop over batches
# evaluate on validation set
# save the model weights

import torch
import torch.nn as nn
import torch.utils.data as data
from model import HybridCNNLSTM
from dataloader import load_dataset, SEQ_LEN

device = "cpu"

# slicing dataset
class StormSurgeDataset(data.Dataset):
    def __init__(self, era5_mm, cora_arr, start_idx):
        self.era5 = era5_mm     # mem-mapped numpy (720, 647, 57, 69)
        self.cora = cora_arr    # ndarray in RAM   (720, nodes)
        self.idxs = start_idx   # 1-D array of sequence start positions

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx]
        x_era5 = self.era5[i : i + SEQ_LEN]     # (T,C,H,W)
        x_cora = self.cora[i : i + SEQ_LEN]     # (T,N)
        y      = self.cora[i + SEQ_LEN]         # (N,)
        return (torch.tensor(x_era5, dtype=torch.float32),
                torch.tensor(x_cora, dtype=torch.float32),
                torch.tensor(y,      dtype=torch.float32))

# load and split dataset by calling dataloader.py
era5_mm, cora, tr_idx, va_idx, te_idx, mask_np = load_dataset()
mask = torch.from_numpy(mask_np).to(device)

train_ds = StormSurgeDataset(era5_mm, cora, tr_idx)
val_ds   = StormSurgeDataset(era5_mm, cora, va_idx)

train_loader = data.DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=2)
val_loader   = data.DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=2)

# initialize the model
model = HybridCNNLSTM(
    era5_channels = era5_mm.shape[1], 
    zeta_nodes = mask.sum().item()  
).to(device)

# loss and optimizer
criterion = nn.MSELoss() # measures average squared error between prediction and target
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # adapts learning rates per parameter

# training loop
epochs = 3 #change later
for epoch in range(epochs):
    model.train() # set model to training mode
    train_loss = 0.0 #reset training loss tracker
    for x_era5, x_cora, y_true in train_loader:
        x_era5, x_cora, y_true = (
            x_era5.to(device),
            x_cora.to(device),
            y_true.to(device)
        )
        # replace any more nans with zeros
        x_era5 = torch.nan_to_num(x_era5, nan=0.0)
        x_cora = torch.nan_to_num(x_cora, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)

        optimizer.zero_grad() # zero out gradients
        y_pred = model(x_era5, x_cora) # get predictions
        loss = criterion(y_pred, y_true) # compute loss
        loss.backward() # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step() # update weights
        train_loss += loss.item()

    model.eval() # switch to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for x_era5, x_cora, y_true in val_loader:
            x_era5, x_cora, y_true = (
                x_era5.to(device),
                x_cora.to(device),
                y_true.to(device)
            )
            x_era5 = torch.nan_to_num(x_era5, nan=0.0) 
            x_cora = torch.nan_to_num(x_cora, nan=0.0)
            y_true = torch.nan_to_num(y_true, nan=0.0)

            y_pred = model(x_era5, x_cora)
            loss = criterion(y_pred, y_true)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# save model
torch.save(model.state_dict(), "cnn_lstm_model.pth")