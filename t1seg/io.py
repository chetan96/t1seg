"""Data format + loaders.

Canonical on-disk format: one ``.npz`` per slice with
    images        (K, H, W) float   T1-weighted magnitude images
    tis           (K,)      float   inversion times in ms (np.inf allowed, e.g. STONE's TI=inf image)
    masks         (K, H, W) uint8   optional, 1 = LV myocardium
    pixel_spacing (2,)      float   mm
Everything downstream (training, pipeline) reads this format, so to use a
new dataset you only need a converter that writes these files.
"""
from pathlib import Path
import re

import numpy as np


def save_slice(path, images, tis, masks=None, pixel_spacing=(1.4, 1.4)):
    d = dict(images=np.asarray(images, np.float32), tis=np.asarray(tis, np.float64),
             pixel_spacing=np.asarray(pixel_spacing, np.float64))
    if masks is not None:
        d["masks"] = np.asarray(masks, np.uint8)
    np.savez_compressed(path, **d)


def load_slice(path):
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    out.setdefault("pixel_spacing", np.array([1.4, 1.4]))
    return out


def pad_crop(img, size=256):
    """Center pad/crop a (..., H, W) array to (..., size, size)."""
    h, w = img.shape[-2:]
    out = np.zeros(img.shape[:-2] + (size, size), img.dtype)
    sh, sw = max(0, (h - size) // 2), max(0, (w - size) // 2)
    dh, dw = max(0, (size - h) // 2), max(0, (size - w) // 2)
    ch, cw = min(h, size), min(w, size)
    out[..., dh:dh + ch, dw:dw + cw] = img[..., sh:sh + ch, sw:sw + cw]
    return out


def _dicom_ti(ds):
    ti = getattr(ds, "InversionTime", None)
    if ti not in (None, ""):
        return float(ti)
    # Siemens MOLLI often stores it as e.g. "TI 180" in ImageComments
    m = re.search(r"TI\s*[:=]?\s*(\d+(\.\d+)?)", str(getattr(ds, "ImageComments", "")))
    return float(m.group(1)) if m else np.nan


def load_mat(path, verbose=False):
    """Load a .mat file and return its contents as a dict of numpy arrays.

    Works with both MATLAB v5 (scipy) and v7.3/HDF5 (h5py) files.
    If *verbose*, prints each key with dtype and shape.
    """
    path = Path(path)
    try:
        from scipy.io import loadmat
        d = loadmat(str(path), squeeze_me=False)
        out = {k: v for k, v in d.items() if not k.startswith("_")}
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(str(path), "r") as f:
            for k in f:
                v = f[k]
                out[k] = v[()].T if isinstance(v, h5py.Dataset) else v
    if verbose:
        for k, v in out.items():
            v = np.asarray(v)
            print(f"  {k:24s}  {str(v.dtype):10s}  {v.shape}")
    return out


def load_dicom_series(folder):
    """Load the T1-weighted images of ONE slice from a folder of DICOMs.

    Returns dict in the canonical format (no masks), sorted by TI.
    Vendor TI storage varies -- check the returned ``tis`` before trusting it.
    """
    import pydicom

    dss = [pydicom.dcmread(f) for f in sorted(Path(folder).iterdir()) if f.is_file()]
    tis = np.array([_dicom_ti(d) for d in dss])
    order = np.argsort(tis)
    imgs = np.stack([dss[i].pixel_array.astype(np.float32) for i in order])
    return dict(images=imgs, tis=tis[order],
                pixel_spacing=np.array(dss[0].PixelSpacing, np.float64))
