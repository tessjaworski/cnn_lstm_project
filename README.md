# **Storm Surge Forecasting with CNN-LSTM**

This project trains a hybrid **CNN + LSTM** model to predict coastal storm surge (zeta) using **ERA5** and **CORA** .

---

## **Files**

- `model.py` — Defines the hybrid CNN + LSTM model  
- `dataloader.py` — Loads ERA5 and CORA data from `.nc` files  
- `train.py` — Trains the model and saves weights to `hybrid_model.pth`  
- `evaluate.py` — Evaluates model performance on the test set  

---
