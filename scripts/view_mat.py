"""Visualize arrays from a .mat file, with optional mask overlay.

Handles 2-D, 3-D, and 4-D arrays.  For 4-D MATLAB arrays laid out as
(H, W, channels, slices) — like the T1 mapping dataset — use --channel
to pick a channel index and the script iterates over slices (dim 4).

Examples
--------
# List what's inside a .mat (keys, shapes, dtypes)
python scripts/view_mat.py /v/rx/t1_resize.mat --list

# Show T1w image channel 0 across all slices
python scripts/view_mat.py /v/rx/t1_resize.mat -k t1_resize -c 0

# Show the T1 map (channel 8) with jet colormap
python scripts/view_mat.py /v/rx/t1_resize.mat -k t1_resize -c 8 --cmap jet --vmin 0 --vmax 2000

# Show all 8 T1w images for one slice (--channel 0:8 picks channels, --slices picks slice)
python scripts/view_mat.py /v/rx/t1_resize.mat -k t1_resize -c 0:8 -s 3

# Overlay endo mask (mask channel 0), skip slices with empty masks
python scripts/view_mat.py /v/rx/t1_resize.mat -k t1_resize -c 0 \
    -m endo_epi_masks --mask-file /v/rx/endo_epi_masks.mat --mask-channel 0 \
    --filter-empty

# Overlay both endo (red) and epi (cyan) contours
python scripts/view_mat.py /v/rx/t1_resize.mat -k t1_resize -c 0 \
    -m endo_epi_masks --mask-file /v/rx/endo_epi_masks.mat --mask-channel 0 1 \
    --mask-color r c --filter-empty

# Save instead of showing
python scripts/view_mat.py /v/rx/t1_resize.mat -k t1_resize -c 8 --cmap jet -o t1maps.png
"""
import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from t1seg.io import load_mat


# ── layer renderers ──────────────────────────────────────────────────
# Each renderer draws onto an existing Axes.  New visualisation layers
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


# ── array slicing helpers ────────────────────────────────────────────

def parse_channel_arg(s):
    """Parse '3' -> [3], '0:8' -> [0,1,...,7], '0 3 5' handled by nargs."""
    if ":" in s:
        parts = s.split(":")
        return list(range(int(parts[0]), int(parts[1])))
    return [int(s)]


def extract_panels(arr, channels=None, slices=None):
    """Yield (title, 2-D image) from a 2-D, 3-D, or 4-D array.

    For 4-D (H, W, C, S):
      - channels selects along dim 2, slices selects along dim 3.
      - If one channel given: one panel per slice.
      - If multiple channels and one slice: one panel per channel.
      - If both multiple: grid (title shows both).
    For 3-D: slices selects along dim 0.
    For 2-D: yields the single image.
    """
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 2:
        yield "", arr
        return

    if arr.ndim == 3:
        idxs = slices if slices is not None else list(range(arr.shape[0]))
        for i in idxs:
            yield f"slice {i}", arr[i]
        return

    if arr.ndim == 4:
        # (H, W, C, S) — MATLAB convention
        all_c = channels if channels is not None else list(range(arr.shape[2]))
        all_s = slices if slices is not None else list(range(arr.shape[3]))
        for s in all_s:
            for c in all_c:
                label_parts = []
                if len(all_s) > 1 or slices is not None:
                    label_parts.append(f"slice {s}")
                if len(all_c) > 1 or channels is not None:
                    label_parts.append(f"ch {c}")
                yield "  ".join(label_parts), arr[:, :, c, s]
        return

    raise ValueError(f"expected 2/3/4-D array, got shape {arr.shape}")


def extract_mask_panels(mask_arr, mask_channels, slices=None):
    """Yield list-of-(2-D mask) per panel position, matching extract_panels ordering.

    Each yield is a list because there can be multiple mask channels (endo + epi).
    """
    mask_arr = np.asarray(mask_arr, dtype=float)

    if mask_arr.ndim == 2:
        yield [mask_arr]
        return

    if mask_arr.ndim == 3:
        idxs = slices if slices is not None else list(range(mask_arr.shape[0]))
        for i in idxs:
            yield [mask_arr[i]]
        return

    if mask_arr.ndim == 4:
        all_s = slices if slices is not None else list(range(mask_arr.shape[3]))
        mc = mask_channels if mask_channels is not None else list(range(mask_arr.shape[2]))
        for s in all_s:
            yield [mask_arr[:, :, c, s] for c in mc]
        return


def find_nonempty_slices(mask_arr, mask_channels=None):
    """Return sorted list of slice indices (dim 3) where at least one mask channel is nonzero."""
    mask_arr = np.asarray(mask_arr)
    if mask_arr.ndim < 4:
        return None  # not applicable
    mc = mask_channels if mask_channels is not None else list(range(mask_arr.shape[2]))
    good = []
    for s in range(mask_arr.shape[3]):
        if any(mask_arr[:, :, c, s].any() for c in mc):
            good.append(s)
    return good


# ── main display ─────────────────────────────────────────────────────

def show(image_key, data, channels=None, mask_key=None, mask_data=None,
         mask_channels=None, slices=None, cmap="gray", vmin=None, vmax=None,
         mask_mode="contour", mask_colors=None, filter_empty=False,
         out_path=None):
    """High-level display: image panels with optional mask overlay."""
    img = np.asarray(data[image_key], dtype=float)

    # resolve mask source
    mask_arr = None
    if mask_key:
        src = mask_data if mask_data is not None else data
        mask_arr = np.asarray(src[mask_key], dtype=float)

    # filter empty slices before building panels
    if filter_empty and mask_arr is not None:
        good = find_nonempty_slices(mask_arr, mask_channels)
        if good is not None:
            if slices is not None:
                slices = [s for s in slices if s in good]
            else:
                slices = good
            n_total = mask_arr.shape[3]
            print(f"filter-empty: keeping {len(slices)}/{n_total} slices with masks")

    panels = list(extract_panels(img, channels=channels, slices=slices))

    if not panels:
        print("nothing to display (all slices filtered out?)")
        return

    if mask_colors is None:
        mask_colors = ["r"]

    ncols = min(len(panels), 8)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    # pre-build mask panel list (one entry per image-slice, each entry is a list of 2-D masks)
    mask_panels = None
    if mask_arr is not None:
        mask_panels = list(extract_mask_panels(mask_arr, mask_channels, slices=slices))

    for idx, (title, panel) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        render_image(ax, panel, cmap=cmap, vmin=vmin, vmax=vmax, title=title)

        if mask_panels is not None and idx < len(mask_panels):
            for mi, m2d in enumerate(mask_panels[idx]):
                color = mask_colors[mi % len(mask_colors)]
                if mask_mode == "contour":
                    render_mask_contour(ax, m2d, color=color)
                else:
                    render_mask_overlay(ax, m2d, color=color)

    suptitle = image_key
    if mask_key:
        suptitle += f"  +  {mask_key}"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualize arrays from a .mat file",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat_file", help="path to .mat file")
    ap.add_argument("--list", action="store_true", help="just list keys and exit")
    ap.add_argument("--key", "-k", help="array key to display")
    ap.add_argument("--channel", "-c", nargs="+",
                    help="for 4-D (H,W,C,S): channel index(es) along dim 3 "
                         "(e.g. 0, '0:8', or 0 3 5)")
    ap.add_argument("--slices", "-s", type=int, nargs="+",
                    help="which slices to show (dim 0 for 3-D, dim 4 for 4-D)")
    ap.add_argument("--cmap", default="gray")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--mask", "-m", help="key of mask array to overlay")
    ap.add_argument("--mask-file", help=".mat file containing the mask (default: same file)")
    ap.add_argument("--mask-channel", type=int, nargs="+",
                    help="channel index(es) in the mask array (e.g. 0 for endo, 1 for epi)")
    ap.add_argument("--mask-mode", choices=["contour", "overlay"], default="contour")
    ap.add_argument("--mask-color", nargs="+", default=["r"],
                    help="color(s) for mask channels (e.g. r c for endo=red epi=cyan)")
    ap.add_argument("--filter-empty", action="store_true",
                    help="skip slices where the mask is all zeros")
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

    # parse --channel (supports range syntax like 0:8)
    channels = None
    if args.channel:
        channels = []
        for tok in args.channel:
            channels.extend(parse_channel_arg(tok))

    mask_data = None
    if args.mask and args.mask_file:
        mask_data = load_mat(args.mask_file, verbose=True)
        print()

    show(args.key, data,
         channels=channels,
         mask_key=args.mask, mask_data=mask_data,
         mask_channels=args.mask_channel,
         slices=args.slices, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
         mask_mode=args.mask_mode, mask_colors=args.mask_color,
         filter_empty=args.filter_empty,
         out_path=args.out)


if __name__ == "__main__":
    main()
