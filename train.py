# loads dataset from dataloader.py
# initializes cnn+lstm model from model.py
# define loss function and optimizer
# run the training loop over batches
# evaluate on validation set
# save the model weights

import psutil

def log_mem(stage):
    vm = psutil.virtual_memory()
    print(f"[{stage}] Used {(vm.used/1024**3):.1f} GB / Available {(vm.available/1024**3):.1f} GB")

log_mem("start")

import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from model import CNN_GNN_Hybrid
from dataloader import load_dataset, CORA_PATHS, SEQ_LEN, PRED_LEN
from cora_graph      import load_cora_coordinates, build_edge_index
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# slicing dataset
class StormSurgeDataset(data.Dataset):
    def __init__(self,
                 era5_mm, μ_e5, σ_e5,
                 cora_arr, μ_c, σ_c,
                 start_idx):
        self.era5 = era5_mm     # mem-mapped numpy (720, 647, 57, 69)
        self.μ_e5 = μ_e5
        self.σ_e5 = σ_e5
        self.cora = cora_arr    # ndarray in RAM   (720, nodes)
        self.μ_c  = μ_c
        self.σ_c  = σ_c
        self.idxs = start_idx   # 1-D array of sequence start positions

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx]

        # ERA5 sequence + normalization
        x = self.era5[i : i+SEQ_LEN]
        x = (x - self.μ_e5) / (self.σ_e5 + 1e-6)

        # CORA past sequence
        z = self.cora[i : i+SEQ_LEN]
        z = (z - self.μ_c) / (self.σ_c + 1e-6)

        #CORA future targets
        y = self.cora[i+SEQ_LEN : i+SEQ_LEN+PRED_LEN]
        y = (y - self.μ_c) / (self.σ_c + 1e-6)

        return (torch.from_numpy(x).float(),
                torch.from_numpy(z).float(),
                torch.from_numpy(y).float())


# Build kNN graph over your CORA node coordinates  
log_mem("before load_dataset")
era5_mm, μ_e5, σ_e5, \
cora,   μ_cora, σ_cora, \
tr_idx, va_idx, test_idx, mask_np = load_dataset()
log_mem("after load_dataset")
coords     = load_cora_coordinates(CORA_PATHS[0], mask_np)
edge_index = build_edge_index(coords, k=8).to(device) 
train_ds = StormSurgeDataset(
    era5_mm, μ_e5, σ_e5,
    cora,   μ_cora, σ_cora,
    tr_idx
)
val_ds   = StormSurgeDataset(
    era5_mm, μ_e5, σ_e5,
    cora,   μ_cora, σ_cora,
    va_idx
)

train_loader = data.DataLoader(
    train_ds, batch_size=4,
    shuffle=True,  num_workers=4, pin_memory=True
)
val_loader   = data.DataLoader(
    val_ds,   batch_size=1,
    shuffle=False, num_workers=0, pin_memory=False
)

num_era5_feats = era5_mm.shape[1]
# initialize the model
model = CNN_GNN_Hybrid(
    era5_channels     = era5_mm.shape[1],   # number of ERA5 channels per grid cell
    cnn_hidden        = 4,                 # match your CNN channels
    cnn_lstm_hidden   = 8,
    gcn_hidden        = 4,
    zeta_lstm_hidden  = 4,
    pred_steps        = PRED_LEN
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# loss and optimizer
criterion = nn.MSELoss() # measures average squared error between prediction and target
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4) # adapts learning rates per parameter

patience = 8
lr_reduce_patience = 4
min_delta = 1e-3

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=lr_reduce_patience,
    min_lr=1e-6
)

best_val = float("inf") 
epochs_no_improve = 0  

train_losses = []
val_losses   = []

# training loop
epochs = 10
for epoch in range(epochs):
    model.train() # set model to training mode
    train_loss = 0.0 #reset training loss tracker
    for x_era5, zeta_past, y_true in train_loader:
        # x_era5: [B, SEQ_LEN, C, H, W]
        # zeta_past: [B, SEQ_LEN, N]
        # y_true: [B, PRED_LEN, N]
        x_era5    = torch.nan_to_num(x_era5,   nan=0.0).to(device)
        zeta_past = torch.nan_to_num(zeta_past,nan=0.0).to(device)
        y_true    = torch.nan_to_num(y_true,   nan=0.0).to(device)

        B, T, C, H, W = x_era5.shape
        era5_seq = x_era5


        optimizer.zero_grad()
        # forward through your GCNHybrid model
        y_pred = model(era5_seq, zeta_past, edge_index)  # [B, PRED_LEN, N]
        loss   = criterion(y_pred, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f}")       

    model.eval() # switch to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for x_era5, zeta_past, y_true in val_loader:
            # Move and clean data
            x_era5    = torch.nan_to_num(x_era5,   nan=0.0).to(device)  # [B, T, C, H, W]
            zeta_past = torch.nan_to_num(zeta_past,nan=0.0).to(device)  # [B, T, N]
            y_true    = torch.nan_to_num(y_true,   nan=0.0).to(device)  # [B, PRED_LEN, N]

            # Extract node‐level ERA5 features
            B, T, C, H, W = x_era5.shape
            era5_seq = x_era5

            # Forward pass
            y_pred = model(era5_seq, zeta_past, edge_index) # [B, PRED_LEN, N]
            loss   = criterion(y_pred, y_true)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

    scheduler.step(val_loss) # adjust LR on plateu

    # early stopping logic
    #if best_val - val_loss > min_delta:
     #   best_val = val_loss
     #   epochs_no_improve = 0
     #   torch.save(model.state_dict(), "gnn_model_24h_normalized.pth")  # * save best
   #else:
      #  epochs_no_improve += 1
      #  if epochs_no_improve >= patience:
      #      print(f"No improvement in {patience} epochs → early stopping.")
      #      break

plt.figure(figsize=(6,4))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)  +1), val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Train vs Validation MSE over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150)
print("Saved learning_curve.png")

# save model
torch.save(model.state_dict(), "gnn_model_24h_normalized.pth")