# prestack all ERA5 NetCDF files into one NumPy array
import os
import numpy as np
import xarray as xr
import re
from numpy.lib.format import open_memmap


ERA5_ROOT = "/media/volume/era5_cora_data/era5_gulf_data"
MONTHS    = ["201501", "201502", "201503", "201504", "201505", "201506", "201507","201508","201509","201510","201511","201512",]
OUT_FILE   = "/media/volume/era5_cora_data/stacked_era5_12mo.npy" 
GRID_SHAPE = (57, 69)

def compute_tref(months):
    import calendar
    total_days = sum(calendar.monthrange(int(m[:4]), int(m[4:]))[1] for m in months)
    return total_days * 24

t_ref = compute_tref(MONTHS)

PL_RE  = re.compile(r"an\.pl.*?_\d+_(\w+)\.ll")
SFC_RE = re.compile(r"an\.sfc.*?_\d+_(\w+)\.ll")
VIN_RE = re.compile(r"an\.vinteg.*?_\d+_(\w+)\.ll")


# Get unique variable names from filenames
def list_vars(files, regex):
    return sorted({regex.search(f).group(1).upper() for f in files if regex.search(f)})

def pick_vname(ds, var):
    if var in ds.data_vars:
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

            vals = ds[vname].values
            if vals.ndim == 3:
                vals = vals[:, None, :, :] 

            arrs.append(vals)

    if not arrs:
        return None

    data = np.concatenate(arrs, axis=0)
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
       # Gather all filenames
    all_files = []
    for m in MONTHS:
        d = os.path.join(ERA5_ROOT, f"era5_gulf_cropped_{m}")
        all_files.extend(os.listdir(d))

    pl_vars  = list_vars(all_files, PL_RE)
    sfc_vars = list_vars(all_files, SFC_RE)
    vin_vars = list_vars(all_files, VIN_RE)

    print("Pressure-level vars:", pl_vars)
    print("Surface-level vars:", sfc_vars)
    print("Vert-integrated vars:", vin_vars)

    # Map each group to its loader function
    var_loaders = [ (pl_vars, load_pl_var),
                    (sfc_vars, load_sfc_var),
                    (vin_vars, load_vinteg_var) ]
# First pass: figure out which vars we will actually write
    write_list = []
    channels_per_var = {}
    for var_list, loader in var_loaders:
        for var in var_list:
            data = loader(var)
            if data is None or data.shape[0] != t_ref:
                print(f"[skip-len] {var} has "
                      f"{None if data is None else data.shape[0]} h "
                      f"(expected {t_ref})")
                continue
            n_chan = data.shape[1]
            channels_per_var[var] = n_chan
            write_list.append((var, loader))
            del data

    # Compute total_channels and allocate full array in memory
    total_channels = sum(channels_per_var.values())
    print(f"Total channels to write: {total_channels}")

    full_shape = (t_ref, total_channels, *GRID_SHAPE)
    print(f"Allocating full array of shape {full_shape} in RAM…")
    era5 = np.empty(full_shape, dtype=np.float32)

    # Fill it chunk‐by‐chunk
    chan_idx = 0
    for var, loader in write_list:
        n_chan = channels_per_var[var]
        print(f"Writing {var}: channels {chan_idx}:{chan_idx+n_chan}")
        block = loader(var).astype(np.float32)  # shape (t_ref, n_chan, H, W)
        era5[:, chan_idx:chan_idx+n_chan, :, :] = block
        chan_idx += n_chan
        del block

    # Save once to disk
    print(f"Saving full array to {OUT_FILE} …")
    np.save(OUT_FILE, era5)
    print("Done.")

if __name__ == "__main__":
    main()
