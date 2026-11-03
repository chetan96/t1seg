"""Convert .mat T1 mapping data to per-slice .npz files for the pipeline.

Reads two .mat files:
  - images: (H, W, C, S) — channels 0..N-2 are T1w images, channel N-1 is the T1 map
  - masks:  (H, W, M, S) — M mask channels (endo, epi); slices with all-zero masks are skipped

Writes one .npz per slice in the pipeline's canonical format:
  images        (K, H, W) float32   T1-weighted images (T1 map excluded)
  tis           (K,)      float64   inversion times (dummy if not known)
  masks         (K, H, W) uint8     myocardium = filled(epi) & ~filled(endo), same for all K
  pixel_spacing (2,)      float64
  t1_map        (H, W)    float32   the T1 map from the .mat (for reference)

Examples
--------
# Inspect what's in the files
python scripts/convert_mat.py --images /v/rx/moco_raw_imgs_t1_resize.mat \
    --masks /v/rx/endo_epi_masks.mat --list

# Convert only slices that have ground-truth masks
python scripts/convert_mat.py --images /v/rx/moco_raw_imgs_t1_resize.mat \
    --masks /v/rx/endo_epi_masks.mat --out data/real_slices --filter-empty

# Convert specific slices
python scripts/convert_mat.py --images /v/rx/moco_raw_imgs_t1_resize.mat \
    --masks /v/rx/endo_epi_masks.mat --out data/real_slices --slices 0 1 2 5 10

# Then run CineMA on them:
python scripts/segment.py -c configs/pipeline/cinema_acdc.yaml \
    --inputs data/real_slices/*.npz --out outputs/cinema_acdc
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes

from t1seg.io import load_mat, save_slice


def fill_mask(m):
    """Fill holes in a binary mask (contour -> filled region)."""
    b = m > 0
    if not b.any():
        return b
    return binary_fill_holes(b)


def make_myocardium(endo, epi):
    """Myocardium = filled(epi) & ~filled(endo)."""
    return (fill_mask(epi) & ~fill_mask(endo)).astype(np.uint8)


def find_key(data, hint=None):
    """Find the main data key (skip MATLAB metadata)."""
    keys = [k for k in data if not k.startswith("_")]
    if hint and hint in keys:
        return hint
    if len(keys) == 1:
        return keys[0]
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Convert .mat T1 data to per-slice .npz for the pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help=".mat file with T1w images (H,W,C,S)")
    ap.add_argument("--masks", required=True, help=".mat file with endo/epi masks (H,W,M,S)")
    ap.add_argument("--image-key", help="key in images .mat (auto-detected if one key)")
    ap.add_argument("--mask-key", help="key in masks .mat (auto-detected if one key)")
    ap.add_argument("--out", default="data/real_slices", help="output directory for .npz files")
    ap.add_argument("--list", action="store_true", help="just show file contents and exit")
    ap.add_argument("--filter-empty", action="store_true",
                    help="skip slices where masks are all zero")
    ap.add_argument("--slices", type=int, nargs="+",
                    help="only convert these slice indices (0-based)")
    ap.add_argument("--max-slices", type=int, help="convert at most N slices (after filtering)")
    ap.add_argument("--t1w-channels", type=int, nargs="+",
                    help="which channels are T1w images (default: 0..C-2, last is T1 map)")
    ap.add_argument("--t1map-channel", type=int,
                    help="which channel is the T1 map (default: last)")
    ap.add_argument("--endo-channel", type=int, default=0,
                    help="mask channel for endocardial contour (default: 0)")
    ap.add_argument("--epi-channel", type=int, default=1,
                    help="mask channel for epicardial contour (default: 1)")
    ap.add_argument("--pixel-spacing", type=float, default=1.0,
                    help="pixel spacing in mm (default: 1.0)")
    args = ap.parse_args()

    print("Loading images:", args.images)
    img_data = load_mat(args.images, verbose=True)
    print()
    print("Loading masks:", args.masks)
    mask_data = load_mat(args.masks, verbose=True)
    print()

    if args.list:
        return

    img_key = find_key(img_data, args.image_key)
    mask_key = find_key(mask_data, args.mask_key)
    if img_key is None:
        print(f"Multiple keys in images file — pick one with --image-key: {list(img_data.keys())}")
        sys.exit(1)
    if mask_key is None:
        print(f"Multiple keys in masks file — pick one with --mask-key: {list(mask_data.keys())}")
        sys.exit(1)

    imgs = np.asarray(img_data[img_key], dtype=np.float32)
    masks = np.asarray(mask_data[mask_key], dtype=np.float32)
    print(f"Images key='{img_key}' shape={imgs.shape}")
    print(f"Masks  key='{mask_key}' shape={masks.shape}")

    if imgs.ndim != 4:
        print(f"Expected 4-D images (H,W,C,S), got {imgs.ndim}-D")
        sys.exit(1)
    if masks.ndim != 4:
        print(f"Expected 4-D masks (H,W,M,S), got {masks.ndim}-D")
        sys.exit(1)

    H, W, C, S = imgs.shape
    _, _, M, S_mask = masks.shape
    if S != S_mask:
        print(f"Slice count mismatch: images have {S}, masks have {S_mask}")
        sys.exit(1)

    t1w_chs = args.t1w_channels or list(range(C - 1))
    t1map_ch = args.t1map_channel if args.t1map_channel is not None else C - 1
    K = len(t1w_chs)
    print(f"\nSlices: {S}  |  T1w channels: {t1w_chs}  |  T1 map channel: {t1map_ch}")
    print(f"Mask channels: endo={args.endo_channel}, epi={args.epi_channel} (of {M} total)")
    print(f"Pixel spacing: {args.pixel_spacing} mm")

    # Determine which slices to convert
    all_slices = args.slices if args.slices is not None else list(range(S))

    if args.filter_empty:
        nonempty = []
        for s in all_slices:
            endo = masks[:, :, args.endo_channel, s]
            epi = masks[:, :, args.epi_channel, s]
            if endo.any() or epi.any():
                nonempty.append(s)
        print(f"Filter: {len(nonempty)}/{len(all_slices)} slices have non-empty masks")
        all_slices = nonempty

    if args.max_slices and len(all_slices) > args.max_slices:
        all_slices = all_slices[:args.max_slices]
        print(f"Limiting to first {args.max_slices} slices")

    if not all_slices:
        print("No slices to convert.")
        sys.exit(0)

    # Dummy inversion times (not stored in the .mat)
    dummy_tis = np.arange(1, K + 1, dtype=np.float64) * 200.0
    print(f"Using dummy TIs (not in .mat): {dummy_tis.tolist()}")
    print(f"(Override in the pipeline with real TIs if you have them)\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = np.array([args.pixel_spacing, args.pixel_spacing], dtype=np.float64)

    written = 0
    for s in all_slices:
        t1w = imgs[:, :, t1w_chs, s]            # (H, W, K)
        t1w = np.transpose(t1w, (2, 0, 1))      # (K, H, W)
        t1_map = imgs[:, :, t1map_ch, s]         # (H, W)

        endo = masks[:, :, args.endo_channel, s]
        epi = masks[:, :, args.epi_channel, s]
        myo = make_myocardium(endo, epi)

        # Broadcast to all K frames (same anatomy, different contrast)
        myo_k = np.broadcast_to(myo[None], (K, H, W)).copy()

        fname = out / f"slice_{s:04d}.npz"
        np.savez_compressed(
            fname,
            images=t1w,
            tis=dummy_tis,
            masks=myo_k.astype(np.uint8),
            pixel_spacing=sp,
            t1_map=t1_map.astype(np.float32),
        )
        n_myo = int(myo.sum())
        written += 1
        print(f"  slice {s:3d} -> {fname.name}  "
              f"myo_pixels={n_myo}  img_range=[{t1w.min():.0f}, {t1w.max():.0f}]")

    print(f"\nWrote {written} files to {out}/")
    print(f"\nNext steps:")
    print(f"  # Run CineMA ACDC baseline:")
    print(f"  python scripts/segment.py -c configs/pipeline/cinema_acdc.yaml \\")
    print(f"      --inputs {out}/*.npz --out outputs/cinema_acdc")
    print(f"  # Run CineMA M&Ms baseline:")
    print(f"  python scripts/segment.py -c configs/pipeline/cinema_mnms.yaml \\")
    print(f"      --inputs {out}/*.npz --out outputs/cinema_mnms")


if __name__ == "__main__":
    main()
