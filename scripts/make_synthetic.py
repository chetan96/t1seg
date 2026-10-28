"""Write synthetic slices in the canonical .npz format.

    python scripts/make_synthetic.py --out data/synthetic --n-train 60 --n-val 15 --n-test 5
"""
import argparse
from pathlib import Path

import numpy as np

from t1seg.io import save_slice
from t1seg.synthetic import make_slice, molli_tis, stone_tis

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic")
ap.add_argument("--n-train", type=int, default=60)
ap.add_argument("--n-val", type=int, default=15)
ap.add_argument("--n-test", type=int, default=5)
ap.add_argument("--scheme", choices=["stone", "molli"], default="stone")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
    d = Path(args.out) / split
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rr = rng.uniform(700, 1200)
        tis = stone_tis(rr) if args.scheme == "stone" else molli_tis(rr)
        imgs, masks, tis, t1 = make_slice(rng, tis)
        save_slice(d / f"slice_{i:04d}.npz", imgs, tis, masks)
        np.savez_compressed(d / f"slice_{i:04d}_t1true.npz", t1=t1)
    print(f"{split}: {n} slices -> {d}")
