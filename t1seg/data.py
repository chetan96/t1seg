"""PyTorch dataset: every (T1w image, mask) pair from a folder of slice .npz files."""
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from .io import load_slice, pad_crop
from .registry import DATASETS

NORMALIZERS = {
    "zscore": lambda x: (x - x.mean()) / (x.std() + 1e-6),          # paper
    "minmax": lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6),
    "percentile": lambda x: np.clip((x - np.percentile(x, 1)) / (np.percentile(x, 99) - np.percentile(x, 1) + 1e-6), 0, 1),
}


def normalize(img, method="zscore"):
    return NORMALIZERS[method](img.astype(np.float32)).astype(np.float32)


def elastic(img, mask, rng, alpha=30.0, sigma=6.0):
    h, w = img.shape
    dy = ndi.gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma) * alpha
    dx = ndi.gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma) * alpha
    yy, xx = np.mgrid[:h, :w]
    coords = [yy + dy, xx + dx]
    return (ndi.map_coordinates(img, coords, order=1, mode="reflect"),
            ndi.map_coordinates(mask.astype(float), coords, order=1, mode="reflect") > 0.5)


@DATASETS.register("npz_slices")
class T1wSegDataset(Dataset):
    """Every (T1w image, mask) pair from slice .npz files (see io.py).
    Paper augmentation: translation p=.95, mirroring p=.95, elastic p=.5, no intensity aug."""

    def __init__(self, files=None, root=None, size=256, augment=False, normalization="zscore",
                 p_translate=0.95, p_mirror=0.95, p_elastic=0.5, max_shift=20, seed=0):
        if files is None:
            files = sorted(Path(root).glob("*.npz"))
            files = [f for f in files if "_t1true" not in f.name]
        self.size, self.augment, self.normalization = size, augment, normalization
        self.p = (p_translate, p_mirror, p_elastic)
        self.max_shift = max_shift
        self.rng = np.random.default_rng(seed)
        self.items = []  # (file, k) -- slices loaded lazily and cached
        self._cache = {}
        for f in files:
            n = np.load(f, mmap_mode="r")["tis"].shape[0]
            self.items += [(str(f), k) for k in range(n)]

    @classmethod
    def from_dir(cls, folder, **kw):
        return cls(root=folder, **kw)

    def __len__(self):
        return len(self.items)

    def _get(self, f):
        if f not in self._cache:
            d = load_slice(f)
            self._cache[f] = (pad_crop(d["images"], self.size), pad_crop(d["masks"], self.size))
        return self._cache[f]

    def __getitem__(self, i):
        f, k = self.items[i]
        imgs, masks = self._get(f)
        img, m = normalize(imgs[k], self.normalization), masks[k].astype(bool)
        if self.augment:
            r = self.rng
            if r.random() < self.p[0]:
                sy, sx = r.integers(-self.max_shift, self.max_shift + 1, 2)
                img = np.roll(img, (sy, sx), (0, 1))
                m = np.roll(m, (sy, sx), (0, 1))
            if r.random() < self.p[1]:
                ax = int(r.integers(0, 2))
                img, m = np.flip(img, ax), np.flip(m, ax)
            if r.random() < self.p[2]:
                img, m = elastic(img, m, r)
        return (torch.from_numpy(np.ascontiguousarray(img, np.float32))[None],
                torch.from_numpy(np.ascontiguousarray(m, np.int64)))
