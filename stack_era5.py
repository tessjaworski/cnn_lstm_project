# prestack all ERA5 NetCDF files into one NumPy array
import os
import numpy as np
import xarray as xr
import re

ERA5_DIR = "/home/exouser/era5_gulf_cropped_201501"
OUT_FILE   = "stacked_era5.npy" 
TOTAL_HOURS = 720
GRID_SHAPE = (57, 69)

PL_RE  = re.compile(r"an\.pl.*?_\d+_(\w+)\.ll")
SFC_RE = re.compile(r"an\.sfc.*?_\d+_(\w+)\.ll")
VIN_RE = re.compile(r"an\.vinteg.*?_\d+_(\w+)\.ll")

# Get unique variable names from filenames
def list_vars(files, regex):
    return sorted({regex.search(f).group(1).upper() for f in files if regex.search(f)})


# Load all pressure-level files for one variable
def load_pl_var(var):
    arrs = []
    for fname in sorted(os.listdir(ERA5_DIR)):
        if f"an.pl_{var}." in fname:
            ds = xr.open_dataset(os.path.join(ERA5_DIR, fname))
            # find the DataArray name
            vname = next(
                (v for v in ds.data_vars if v.lower() == var.lower()),
                None
            )
            if vname is None:
                continue      # skip if not found
            arrs.append(ds[vname].values)      # (time, level, 57, 69)
    if not arrs:
        return None
    data = np.concatenate(arrs, axis=0)  # (720, levels, 57, 69)
    t, lev, *_ = data.shape
    return data.reshape(t, lev, *GRID_SHAPE)  # keep all levels


# Load one monthly surface variable 
def load_sfc_var(var):
    for fname in os.listdir(ERA5_DIR):
        if f"an.sfc_{var}." in fname and fname.endswith(".nc"):
            ds = xr.open_dataset(os.path.join(ERA5_DIR, fname))
            return ds[var].values[:, None, :, :]  # (720, 1, 57, 69)
    return None

# load vertically integrated variables
def load_vinteg_var(var):
    for fname in os.listdir(ERA5_DIR):
        if f"an.vinteg_{var}." in fname and fname.endswith(".nc"):
            ds = xr.open_dataset(os.path.join(ERA5_DIR, fname))
            return ds[var].values[:, None, :, :]
    return None

def main():
    files = os.listdir(ERA5_DIR)

    pl_vars  = list_vars(files,  PL_RE)
    sfc_vars = list_vars(files, SFC_RE)
    vin_vars = list_vars(files, VIN_RE)

    print("\n Loading pressure-level vars:", ", ".join(pl_vars))
    print("Loading surface-level vars  :", ", ".join(sfc_vars))
    print(" Loading vert-integrated vars:", ", ".join(vin_vars))

    stacks = []

    # pressure-level
    for var in pl_vars:
        data = load_pl_var(var)
        if data is not None:
            stacks.append(data.reshape(720, -1, *GRID_SHAPE))  # flatten levels -> channels

    # surface
    for var in sfc_vars:
        data = load_sfc_var(var)
        if data is not None:
            stacks.append(data)

    # vertically integrated
    for var in vin_vars:
        data = load_vinteg_var(var)
        if data is not None:
            stacks.append(data)

    # concatenate along channel axis
    era5 = np.concatenate(stacks, axis=1)  # (720, total_channels, 57, 69)
    print("\nFinal ERA5 array shape:", era5.shape)

    # save as memory-mappable .npy
    np.save(OUT_FILE, era5.astype(np.float32))
    print(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()