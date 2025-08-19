# **Storm Surge Forecasting with CNN-LSTM**

This project trains a hybrid **CNN + LSTM** model to predict coastal storm surge (zeta) using **ERA5** and **CORA** .

---

## **Files**

- `model.py` — Defines the hybrid CNN + LSTM model  
- `dataloader.py` — Loads ERA5 and CORA data from `.nc` files  
- `train.py` — Trains the model and saves weights to `cnn_lstm_model.pth`  
- `evaluate.py` — Evaluates model performance on the test set
- `stack_era5.py`— Preprocesses ERA5 data into ml ready format. Output is a .npy file that the dataloader can memory-map directly.
- `cora_graph.py`— Constructs a graph representation of the CORA dataset. Produces adjacency information that is fed into the graph neural network (GNN).

---
