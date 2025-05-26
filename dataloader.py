# load preprocessed ERA5 and CORA numpy arrays
# slide a window over time to build (x,y) training sequences
# split into train, val, and test
#return Pytorch dataset

import os
import numpy as np
import xarray as xr

ERA5_PATH = "/home/exouser/stacked_era5.npy"
CORA_PATH = "/home/exouser/Jan2015_cropped.nc"
SEQ_LENGTH = 10 # past 10 hours of data as input
PREDICT_LENGTH = 1 # predict 1 hour into the future
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

def load_era5():
    return np.load(ERA5_PATH, mmap_mode="r")   # shape (720, C, 57, 69)

def load_cora():
    ds = xr.open_dataset(CORA_PATH)
    # transposes to (time, nodes) so it aligns with ERA5
    zeta = ds["zeta"].transpose("time", "nodes")
    return zeta.values  # shape: (time, 585869)

# create sequences
def build_sequences(era5, cora):
    # initialize empty lists to store data pairs
    X_era5, X_cora, y = [], [], []
    #slides a time window from hour 0 to last possible training point
    for i in range(len(era5) - SEQ_LENGTH - PREDICT_LENGTH + 1):
        # extracts past 10 hours of ERA5 and CORA and the next hour of CORA
        era5_seq = era5[i:i+SEQ_LENGTH]  # (T, C, 57, 69)
        cora_seq = cora[i:i+SEQ_LENGTH]  # (T, nodes)
        target = cora[i+SEQ_LENGTH]      # (nodes,)
        # appends each sequence to the training list
        X_era5.append(era5_seq)
        X_cora.append(cora_seq)
        y.append(target)
    return np.array(X_era5), np.array(X_cora), np.array(y) # converts lists to numpy arrays

# split into train, val, test
def split_dataset(X_era5, X_cora, y):
    total = len(X_era5)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    # train is first 70%, val is next 15%, test is last 15%
    return (
        X_era5[:train_end], X_cora[:train_end], y[:train_end],
        X_era5[train_end:val_end], X_cora[train_end:val_end], y[train_end:val_end],
        X_era5[val_end:], X_cora[val_end:], y[val_end:]
    )

# main function that will be called in train.py
def load_dataset():
    era5 = load_era5()
    cora = load_cora()
    X_era5, X_cora, y = build_sequences(era5, cora)
    return split_dataset(X_era5, X_cora, y)
