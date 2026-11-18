"""Standalone inference: checkpoint + images -> masks + overlay PNGs.

Unlike segment.py (which needs a pipeline config), this script only needs a
checkpoint and input .npz files.  It runs the trained model, applies optional
post-processing, computes Dice against ground truth when available, and writes
per-slice mask .npz files plus overlay visualisations.

Examples
--------
# Basic — just the model, no post-processing
python scripts/inference.py --checkpoint runs/unet_paper/best.pt \
    --inputs data/synthetic/test/*.npz --out outputs/inference

# With post-processing (paper chain: largest component + shape QC + refine)
python scripts/inference.py --checkpoint runs/unet_paper/best.pt \
    --inputs data/synthetic/test/*.npz --out outputs/inference --postprocess

# Real data converted from .mat
python scripts/inference.py --checkpoint runs/finetune_real/best.pt \
    --inputs data/real_slices/*.npz --out outputs/real_inference --postprocess

# Override device
python scripts/inference.py --checkpoint runs/unet_paper/best.pt \
    --inputs data/synthetic/test/*.npz --device cpu

# Run full pipeline (segment -> align -> fit -> T1 stats)
python scripts/inference.py --checkpoint runs/unet_paper/best.pt \
    --inputs data/synthetic/test/*.npz --out outputs/full --full-pipeline
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from t1seg.analysis import dice
from t1seg.data import normalize
from t1seg.device import resolve_device
from t1seg.io import load_slice, pad_crop
from t1seg.models import load_checkpoint
from t1seg.postprocess import postprocess_series


@torch.no_grad()
def predict(model, images, device, size=256, normalization="zscore", batch_size=16):
    """Run model on (K, H, W) images -> (K, H, W) uint8 masks."""
    imgs = pad_crop(np.asarray(images, np.float32), size)
    x = np.stack([normalize(im, normalization) for im in imgs])
    x = torch.from_numpy(x)[:, None]
    parts = []
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size].to(device))
        parts.append(logits.argmax(1).cpu())
    return torch.cat(parts).numpy().astype(np.uint8)


def save_overlay(images, masks, gt, ok, tis, path):
    """Save an overlay PNG: images with predicted (red/yellow) and GT (green) contours."""
    K = len(images)
    fig, axes = plt.subplots(1, K, figsize=(1.8 * K, 2.2))
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        ax.imshow(images[k], cmap="gray")
        if gt is not None:
            ax.contour(gt[k], [0.5], colors="lime", linewidths=0.5)
        if masks[k].any():
            color = "r" if (ok is None or ok[k]) else "y"
            ax.contour(masks[k], [0.5], colors=color, linewidths=0.5)
        ti_str = f"TI={tis[k]:.0f}" if np.isfinite(tis[k]) else "TI=inf"
        ax.set_title(ti_str, fontsize=7)
        ax.axis("off")
    fig.suptitle("red=pred (yellow=failed QC)  green=GT", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Run trained model inference on .npz slices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    ap.add_argument("--inputs", nargs="+", required=True, help=".npz slice files")
    ap.add_argument("--out", default="outputs/inference", help="output directory")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--size", type=int, default=256, help="pad/crop size")
    ap.add_argument("--normalization", default="zscore", choices=["zscore", "minmax", "percentile"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--postprocess", action="store_true",
                    help="apply paper post-processing (largest component + shape QC + refine)")
    ap.add_argument("--full-pipeline", action="store_true",
                    help="run full pipeline: segment -> align -> fit -> T1 stats")
    ap.add_argument("--no-overlay", action="store_true", help="skip overlay PNG generation")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    model, ck = load_checkpoint(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {ck.get('model_cfg', {}).get('name', '?')} ({n_params:.2f}M params) on {device}")
    if "val_dice" in ck:
        print(f"Checkpoint val DSC: {ck['val_dice']:.3f} (epoch {ck.get('epoch', '?')})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []

    for f in args.inputs:
        name = Path(f).stem
        d = load_slice(f)
        images, tis = d["images"], d["tis"]
        pixel_spacing = d.get("pixel_spacing", np.array([1.4, 1.4]))

        masks = predict(model, images, device, args.size, args.normalization, args.batch_size)
        ok = None

        if args.postprocess:
            masks, ok = postprocess_series(masks, tis, pixel_spacing)

        if args.full_pipeline:
            from t1seg.pipeline import Pipeline
            pipe = Pipeline.from_config({
                "segmenter": {"name": "torch", "checkpoint": args.checkpoint},
                "device": args.device,
            })
            res = pipe(images, tis, pixel_spacing, masks=d.get("masks"))
            row = dict(file=f, status=res["status"], n_ok=int(res["ok"].sum()))
            if res["status"] == "ok":
                row.update(res["stats"])
                np.savez_compressed(out / f"{name}_t1map.npz",
                                    t1_map=res["t1_map"], midwall=res["midwall"])
            masks, ok = res["masks"], res["ok"]
        else:
            row = dict(file=f, n_ok=int(ok.sum()) if ok is not None else int((masks > 0).any(-1).any(-1).sum()))

        np.savez_compressed(out / f"{name}_masks.npz", masks=masks,
                            ok=ok if ok is not None else np.ones(len(masks), bool))

        gt = pad_crop(d["masks"], args.size) if "masks" in d else None
        if gt is not None:
            dsc_per_frame = [dice(masks[k], gt[k]) for k in range(len(gt))]
            row["dice_mean"] = float(np.mean(dsc_per_frame))
            row["dice_std"] = float(np.std(dsc_per_frame))

        if not args.no_overlay:
            padded = pad_crop(images, args.size)
            save_overlay(padded, masks, gt, ok, tis, out / f"{name}.png")

        summary.append(row)
        compact = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
        print(json.dumps(compact))

    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    if any("dice_mean" in r for r in summary):
        dices = [r["dice_mean"] for r in summary if "dice_mean" in r]
        print(f"\nOverall Dice: {np.mean(dices):.4f} +/- {np.std(dices):.4f} (n={len(dices)} slices)")
    print(f"Results saved to {out}/")


if __name__ == "__main__":
    main()
