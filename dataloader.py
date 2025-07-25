# load preprocessed ERA5 and CORA numpy arrays
# slide a window over time to build (x,y) training sequences
# split into train, val, and test
#return Pytorch dataset

import numpy as np
import xarray as xr
from pathlib import Path
import torch
from torch.utils.data import Dataset


ERA5_PATH = "/home/exouser/stacked_era5_2mo.npy"
CORA_PATHS = [
    "/media/volume/era5_cora_data/Jan2015_cropped.nc",
    "/media/volume/era5_cora_data/Feb2015_cropped.nc"
]
SEQ_LEN = 10 # past 10 hours of data as input
PRED_LEN = 24 # predict 24 hour into the future
TRAIN_FR = 0.7
VAL_FR = 0.15


def make_full_cora_mask():
    masks = []
    for p in CORA_PATHS:
        ds = xr.open_dataset(p)
        z = ds["zeta"].transpose("time","nodes").values
        masks.append(~np.any(np.isnan(z), axis=0))
    # only keep nodes valid in *all* months
    full_mask = np.logical_and.reduce(masks)
    np.save("zeta_full_mask.npy", full_mask)
    return full_mask

full_mask = make_full_cora_mask()


class CORADataset(Dataset):
    def __init__(self, X, Y):
        # X: numpy array [N_samples, T, num_nodes, num_feats]
        # Y: numpy array [N_samples, num_nodes]
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

def load_era5():
    return np.load(ERA5_PATH, mmap_mode="r")       # shape will be (1416, channels, H, W)

def load_cora():
    arrays = []
    for p in CORA_PATHS:
        ds = xr.open_dataset(p)
        z = ds["zeta"].transpose("time","nodes").values.astype(np.float32)
        # keep only the always-valid nodes
        arrays.append(z[:, full_mask])
    cora = np.concatenate(arrays, axis=0)  # now shape (sum_hours, N_valid)
    return np.nan_to_num(cora, nan=0.0)


# create sequences
def make_indices(T: int):
    last_valid_start = T - SEQ_LEN - PRED_LEN      # inclusive
    idx_all = np.arange(last_valid_start + 1)      # e.g. 0 … 709

    n_train = int(TRAIN_FR * len(idx_all))
    n_val   = int(VAL_FR   * len(idx_all))

    train_idx = idx_all[:n_train]
    val_idx   = idx_all[n_train:n_train + n_val]
    test_idx  = idx_all[n_train + n_val:]
    return train_idx, val_idx, test_idx


# main function that will be called in train.py
def load_dataset():
     # load raw data
    era5_mm = load_era5()          # (744, 647, 57, 69)
    cora    = load_cora()          # (720, nodes)

    # align by trimming ERA-5 to CORA’s 30-day span (720 h)
    L = len(cora) 
    era5_mm = era5_mm[:L]       

    #  build split indices
    train_idx, val_idx, test_idx = make_indices(L)

    #normalization
    e5_train = era5_mm[train_idx]          # shape: (n_train, C, H, W)
    μ_era5   = e5_train.mean(axis=(0,2,3), keepdims=True)
    σ_era5   = e5_train.std (axis=(0,2,3), keepdims=True)
    c_train  = cora[train_idx]             # shape: (n_train, N)
    μ_cora   = c_train.mean(axis=0, keepdims=True)
    σ_cora   = c_train.std (axis=0, keepdims=True)
    era5_mm = (era5_mm - μ_era5) / (σ_era5 + 1e-6)
    cora    = (cora    - μ_cora) / (σ_cora  + 1e-6)

    # return everything to train.py
    return era5_mm, cora, train_idx, val_idx, test_idx, full_mask, μ_cora, σ_cora


if __name__ == "__main__":
    era5_mm, cora, tr, va, te, full_mask = load_dataset() 
    print("ERA-5 slice :", era5_mm.shape, era5_mm.dtype)
    print("CORA        :", cora.shape,    cora.dtype)
    print("splits      :", len(tr), len(va), len(te))