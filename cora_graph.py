import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
import torch


#load CORA coordinates and return an array of coordinates
def load_cora_coordinates(cora_nc_path, mask):
    ds = xr.open_dataset(cora_nc_path)
    lats = ds["lat"].values 
    lons = ds["lon"].values
    coords = np.stack([lats, lons], axis=1)[mask]
    return coords

# with the array of coordinates, build an edge_index
def build_edge_index(coords, k=8):
    tree = cKDTree(coords)
    dist, idxs = tree.query(coords, k=k + 1)
    src = np.repeat(np.arange(len(coords)), k)
    dst = idxs[:, 1 : k + 1].reshape(-1)
    edge_index = torch.tensor(
        np.stack([src, dst], axis=0), dtype=torch.long
    )  # shape: (2, N*k)
    return edge_index