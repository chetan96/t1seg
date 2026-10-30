"""Visualize arrays from a .mat file, with optional mask overlay.

Examples
--------
# List what's inside a .mat
python scripts/view_mat.py data/raw/case01.mat --list

# Show a single 2-D array
python scripts/view_mat.py data/raw/case01.mat --key T1map

# Show a 3-D stack (each slice along axis 0 gets its own panel)
python scripts/view_mat.py data/raw/case01.mat --key images

# Pick specific slices from a 3-D stack
python scripts/view_mat.py data/raw/case01.mat --key images --slices 0 3 5

# Overlay a mask
python scripts/view_mat.py data/raw/case01.mat --key images --mask myocardium

# Overlay a mask from a different .mat file
python scripts/view_mat.py data/raw/case01.mat --key images --mask segmentation --mask-file data/raw/case01_seg.mat

# Save instead of showing
python scripts/view_mat.py data/raw/case01.mat --key T1map -o fig.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from t1seg.io import load_mat


# ── layer renderers ──────────────────────────────────────────────────
# Each renderer draws onto an existing Axes. New visualisation layers
# (contours, ROI highlights, AHA segments) can be added here.

def render_image(ax, img, cmap="gray", vmin=None, vmax=None, title=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_mask_contour(ax, mask, color="r", linewidth=0.8):
    binary = (np.asarray(mask) > 0).astype(float)
    if binary.any():
        ax.contour(binary, levels=[0.5], colors=[color], linewidths=[linewidth])


def render_mask_overlay(ax, mask, color="r", alpha=0.25):
    rgba = np.zeros(mask.shape + (4,))
    c = matplotlib.colors.to_rgb(color)
    m = mask > 0
    rgba[m] = (*c, alpha)
    ax.imshow(rgba, origin="upper")


# ── layout helpers ───────────────────────────────────────────────────

def _to_slices(arr, indices=None):
    """Yield (title, 2-D array) for each slice to display."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        yield "", arr
        return
    if arr.ndim == 3:
        idxs = indices if indices is not None else range(arr.shape[0])
        for i in idxs:
            yield f"slice {i}", arr[i]
        return
    raise ValueError(f"expected 2-D or 3-D array, got shape {arr.shape}")


def show(image_key, data, mask_key=None, mask_data=None,
         slices=None, cmap="gray", vmin=None, vmax=None,
         mask_mode="contour", mask_color="r", out_path=None):
    """High-level display: image panels with optional mask overlay."""
    img = np.asarray(data[image_key], dtype=float)
    panels = list(_to_slices(img, slices))

    mask_arr = None
    if mask_key:
        src = mask_data if mask_data is not None else data
        mask_arr = np.asarray(src[mask_key])

    ncols = min(len(panels), 8)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, (title, panel) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        render_image(ax, panel, cmap=cmap, vmin=vmin, vmax=vmax, title=title)

        if mask_arr is not None:
            m_slices = list(_to_slices(mask_arr, slices))
            if idx < len(m_slices):
                m = m_slices[idx][1]
                if mask_mode == "contour":
                    render_mask_contour(ax, m, color=mask_color)
                else:
                    render_mask_overlay(ax, m, color=mask_color)

    fig.suptitle(f"{image_key}" + (f"  +  {mask_key}" if mask_key else ""),
                 fontsize=11)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Visualize arrays from a .mat file")
    ap.add_argument("mat_file", help="path to .mat file")
    ap.add_argument("--list", action="store_true", help="just list keys and exit")
    ap.add_argument("--key", "-k", help="array key to display")
    ap.add_argument("--slices", "-s", type=int, nargs="+",
                    help="for 3-D arrays: which slices along axis 0 (default: all)")
    ap.add_argument("--cmap", default="gray")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--mask", "-m", help="key of mask array to overlay")
    ap.add_argument("--mask-file", help=".mat file containing the mask (default: same file)")
    ap.add_argument("--mask-mode", choices=["contour", "overlay"], default="contour")
    ap.add_argument("--mask-color", default="r")
    ap.add_argument("-o", "--out", help="save figure to file instead of showing")
    args = ap.parse_args()

    data = load_mat(args.mat_file, verbose=True)
    print()

    if args.list:
        return

    if not args.key:
        keys = list(data.keys())
        if len(keys) == 1:
            args.key = keys[0]
            print(f"auto-selected key: {args.key}\n")
        else:
            print(f"multiple keys found — pick one with --key: {keys}")
            sys.exit(1)

    mask_data = None
    if args.mask and args.mask_file:
        mask_data = load_mat(args.mask_file)

    show(args.key, data,
         mask_key=args.mask, mask_data=mask_data,
         slices=args.slices, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
         mask_mode=args.mask_mode, mask_color=args.mask_color,
         out_path=args.out)


if __name__ == "__main__":
    main()
