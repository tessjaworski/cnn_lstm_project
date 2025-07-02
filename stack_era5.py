# prestack all ERA5 NetCDF files into one NumPy array
import os
import numpy as np
import xarray as xr
import re

ERA5_ROOT = "/media/volume/era5_cora_data/era5_gulf_data"
MONTHS    = ["201501", "201502"]
OUT_FILE   = "stacked_era5_2mo.npy" 
GRID_SHAPE = (57, 69)
t_ref = 1416   

PL_RE  = re.compile(r"an\.pl.*?_\d+_(\w+)\.ll")
SFC_RE = re.compile(r"an\.sfc.*?_\d+_(\w+)\.ll")
VIN_RE = re.compile(r"an\.vinteg.*?_\d+_(\w+)\.ll")

def accept(data, var):
    global t_ref
    if data.shape[0] != t_ref:
        print(f"[skip-len] {var} has {data.shape[0]} h (expected {t_ref})")
        return False
    return True

# Get unique variable names from filenames
def list_vars(files, regex):
    return sorted({regex.search(f).group(1).upper() for f in files if regex.search(f)})

def pick_vname(ds, var):
    if var in ds.data_vars:                 # already UPPER
        return var
    low = var.lower()
    return next((v for v in ds.data_vars if v.lower() == low), None)


# Load all pressure-level files for one variable
def load_pl_var(var):
    pat = re.compile(rf"an\.pl.*?_\d+_{var.lower()}\.ll.*\.nc$", re.IGNORECASE)
    arrs = []
    for m in MONTHS:
        d = os.path.join(ERA5_ROOT, f"era5_gulf_cropped_{m}")
        for fname in os.listdir(d):
            if not pat.search(fname): 
                continue
            fpath = os.path.join(d, fname)
            try:
                ds = xr.open_dataset(fpath, engine="netcdf4")
            except Exception as e:
                print(f"[skip] {fname} – {e}")
                continue
            vname = pick_vname(ds, var)
            if vname is None:
                print(f"[warn] {var} missing in {fname}")
                continue

            vals = ds[vname].values     # (24, L, 57, 69)  *or*  (24, 57, 69)
            if vals.ndim == 3:   # single-level field
                vals = vals[:, None, :, :] 

            arrs.append(vals)

    if not arrs:
        return None

    data = np.concatenate(arrs, axis=0)   # (720, L, 57, 69)
    return data  


# Load one monthly surface variable 
def load_sfc_var(var):
    pat = re.compile(rf"an\.sfc.*?_\d+_{var.lower()}\.ll.*\.nc$", re.IGNORECASE)
    arrs = []
    for m in MONTHS:
        d = os.path.join(ERA5_ROOT, f"era5_gulf_cropped_{m}")
        for fname in os.listdir(d):
            if not pat.search(fname): 
                continue
            ds = xr.open_dataset(os.path.join(d, fname))
            vname = pick_vname(ds, var)
            if not vname:
                continue
            # surface vars always have shape (T, H, W), so add a channel dim
            arrs.append(ds[vname].values[:, None, :, :])
    if not arrs:
        return None
    return np.concatenate(arrs, axis=0)


# load vertically integrated variables
def load_vinteg_var(var):
    pat = re.compile(rf"an\.vinteg.*?_\d+_{var.lower()}\.ll.*\.nc$", re.IGNORECASE)
    arrs = []
    for m in MONTHS:
        d = os.path.join(ERA5_ROOT, f"era5_gulf_cropped_{m}")
        for fname in os.listdir(d):
            if not pat.search(fname): 
                continue
            ds = xr.open_dataset(os.path.join(d, fname))
            vname = pick_vname(ds, var)
            if not vname:
                continue
            arrs.append(ds[vname].values[:, None, :, :])
    if not arrs:
        return None
    return np.concatenate(arrs, axis=0)

def main():
    all_files = []
    for m in MONTHS:
        d = os.path.join(ERA5_ROOT, f"era5_gulf_cropped_{m}")
        for fname in os.listdir(d):
            all_files.append(fname)

    pl_vars  = list_vars(all_files,  PL_RE)
    sfc_vars = list_vars(all_files, SFC_RE)
    vin_vars = list_vars(all_files, VIN_RE)

    print("\n Loading pressure-level vars:", ", ".join(pl_vars))
    print("Loading surface-level vars  :", ", ".join(sfc_vars))
    print(" Loading vert-integrated vars:", ", ".join(vin_vars))

    stacks = []

    # pressure-level
    for var in pl_vars:
        data = load_pl_var(var)
        if data is None or not accept(data, var):
            continue
        if data.ndim == 3:
            data = data[:, None, :, :]
        stacks.append(data.reshape(t_ref, -1, *GRID_SHAPE))

    # surface
    for var in sfc_vars:
        data = load_sfc_var(var)
        if data is not None and accept(data, var):
            stacks.append(data)

    # vertically integrated
    for var in vin_vars:
        data = load_vinteg_var(var)
        if data is not None and accept(data, var):
            stacks.append(data)

    # concatenate along channel axis
    era5 = np.concatenate(stacks, axis=1)  # (720, total_channels, 57, 69)
    print("\nFinal ERA5 array shape:", era5.shape)

    # save as memory-mappable .npy
    np.save(OUT_FILE, era5.astype(np.float32))
    print(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()