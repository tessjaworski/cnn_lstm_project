# load preprocessed ERA5 and CORA numpy arrays
# slide a window over time to build (x,y) training sequences
# split into train, val, and test
#return Pytorch dataset

import numpy as np
import xarray as xr
from pathlib import Path

MASK_PATH = pathlib.Path("zeta_mask.npy")
mask = np.load(MASK_PATH)   

ERA5_PATH = "/home/exouser/stacked_era5.npy"
CORA_PATH = "/home/exouser/Jan2015_cropped.nc"
SEQ_LEN = 10 # past 10 hours of data as input
PRED_LEN = 1 # predict 1 hour into the future
TRAIN_FR = 0.7
VAL_FR = 0.15

def load_era5():
    return np.load(ERA5_PATH, mmap_mode="r")   # shape (744, C, 57, 69)

def load_cora():
    zeta = (
        xr.open_dataset(CORA_PATH)["zeta"]
        .transpose("time", "nodes")
        .values.astype(np.float32)
    )
    zeta = zeta[:, mask]
    zeta = np.nan_to_num(zeta, nan=0.0) # clean residual nans
    return zeta   

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

    # return everything to train.py
    return era5_mm, cora, train_idx, val_idx, test_idx, mask

if __name__ == "__main__":
    era5_mm, cora, tr, va, te, mask = load_dataset() 
    print("ERA-5 slice :", era5_mm.shape, era5_mm.dtype)
    print("CORA        :", cora.shape,    cora.dtype)
    print("splits      :", len(tr), len(va), len(te))