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
from dataloader import load_dataset

# this class allows pytorch to index the dataset sample by sample
# each sample has:
   # ERA5 input sequence (x_era5)
   # CORA zeta input sequence (x_cora)
   # target zeta vector (y)
class StormSurgeDataset(data.Dataset):
    def __init__(self, x_era5, x_cora, y):
        self.x_era5 = torch.tensor(x_era5, dtype=torch.float32)
        self.x_cora = torch.tensor(x_cora, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_era5)

    def __getitem__(self, idx):
        return self.x_era5[idx], self.x_cora[idx], self.y[idx]

# load and split dataset by calling dataloader.py
(X_era5_train, X_cora_train, y_train,
 X_era5_val,   X_cora_val,   y_val,
 X_era5_test,  X_cora_test,  y_test) = load_dataset()

#split dataset into batches
train_dataset = StormSurgeDataset(X_era5_train, X_cora_train, y_train)
val_dataset   = StormSurgeDataset(X_era5_val, X_cora_val, y_val)

train_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = data.DataLoader(val_dataset, batch_size=4, shuffle=False)

# initialize the model
model = HybridCNNLSTM(
    era5_channels=X_era5_train.shape[2],
    zeta_nodes=y_train.shape[1]
)

# loss and optimizer
criterion = nn.MSELoss() # measures average squared error between prediction and target
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # adapts learning rates per parameter

# training loop
epochs = 10
for epoch in range(epochs):
    model.train() # set model to training mode
    train_loss = 0.0 #reset training loss tracker
    for x_era5, x_cora, y_true in train_loader:
        optimizer.zero_grad() # zero out gradients
        y_pred = model(x_era5, x_cora) # get predictions
        loss = criterion(y_pred, y_true) # compute loss
        loss.backward() # compute gradients
        optimizer.step() # update weights
        train_loss += loss.item()

    model.eval() # switch to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for x_era5, x_cora, y_true in val_loader:
            y_pred = model(x_era5, x_cora)
            loss = criterion(y_pred, y_true)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# save model
torch.save(model.state_dict(), "cnn+lstm_model.pth")