"""Fine-tune a pretrained checkpoint on real data and compare Dice before/after.

End-to-end workflow:
  1. Split converted .npz slices into train / val / test
  2. Run the pretrained (baseline) model on the test set
  3. Fine-tune on train+val
  4. Run the fine-tuned model on the same test set
  5. Print per-slice and aggregate Dice comparison

Examples
--------
# Minimal — just point at the data and a pretrained checkpoint
python scripts/finetune_eval.py \
    --data data/real_slices \
    --checkpoint runs/default/best.pt

# Control the split and training
python scripts/finetune_eval.py \
    --data data/real_slices \
    --checkpoint runs/default/best.pt \
    --split 0.6 0.2 0.2 \
    --epochs 20 --lr 1e-4 --batch-size 4

# Skip training — just compare two existing checkpoints on a test set
python scripts/finetune_eval.py \
    --data data/real_slices \
    --checkpoint runs/default/best.pt \
    --finetuned-checkpoint runs/finetune_real/best.pt \
    --skip-train

# Use the SMP encoder instead of the default U-Net
python scripts/finetune_eval.py \
    --data data/real_slices \
    --checkpoint runs/smp_pretrained/best.pt \
    --epochs 15

# Fine-tune CineMA (UK Biobank + ACDC pretrained) — no .pt checkpoint needed
python scripts/finetune_eval.py \
    --data data/real_slices \
    --model-config '{"name":"cinema_ft","dataset":"acdc"}' \
    --epochs 15

# Fine-tune CineMA M&Ms variant
python scripts/finetune_eval.py \
    --data data/real_slices \
    --model-config '{"name":"cinema_ft","dataset":"mnms"}' \
    --epochs 15
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from t1seg.analysis import dice
from t1seg.config import load_config
from t1seg.data import T1wSegDataset
from t1seg.device import resolve_device
from t1seg.io import load_slice, pad_crop
from t1seg.models import build_model, load_checkpoint, init_weights_from, save_checkpoint
from t1seg.train import train, evaluate


def split_files(files, ratios, seed):
    """Randomly split files into (train, val, test) according to ratios."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    n_train = max(1, int(len(files) * ratios[0]))
    n_val = max(1, int(len(files) * ratios[1]))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    if len(test_idx) == 0:
        print("WARNING: no test slices with this split — using val as test")
        test_idx = val_idx
    return ([files[i] for i in train_idx],
            [files[i] for i in val_idx],
            [files[i] for i in test_idx])


def symlink_split(files, dest):
    """Create a directory of symlinks to the original .npz files."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in files:
        (dest / f.name).symlink_to(f.resolve())


@torch.no_grad()
def run_inference(model, files, device, size=256, normalization="zscore"):
    """Run model on a list of .npz files, return per-slice Dice scores."""
    model.eval()
    results = []
    for f in files:
        d = load_slice(str(f))
        images = d["images"]
        gt = d.get("masks")
        if gt is None:
            results.append(dict(file=f.name, dice=float("nan"), has_gt=False))
            continue

        imgs = pad_crop(np.asarray(images, np.float32), size)
        gt_pad = pad_crop(gt, size)

        from t1seg.data import normalize
        x = np.stack([normalize(im, normalization) for im in imgs])
        x = torch.from_numpy(x)[:, None].to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)

        dsc_per_frame = [dice(pred[k], gt_pad[k]) for k in range(len(gt_pad))]
        results.append(dict(
            file=f.name,
            dice=float(np.mean(dsc_per_frame)),
            dice_std=float(np.std(dsc_per_frame)),
            n_frames=len(dsc_per_frame),
            has_gt=True,
        ))
    return results


def print_comparison(baseline, finetuned):
    """Print a side-by-side Dice comparison table."""
    base_by_file = {r["file"]: r for r in baseline}
    fine_by_file = {r["file"]: r for r in finetuned}
    all_files = sorted(set(base_by_file) | set(fine_by_file))

    print(f"\n{'Slice':<25s} {'Baseline':>10s} {'Finetuned':>10s} {'Delta':>10s}")
    print("-" * 58)
    deltas = []
    for f in all_files:
        b = base_by_file.get(f, {}).get("dice", float("nan"))
        ft = fine_by_file.get(f, {}).get("dice", float("nan"))
        d = ft - b if np.isfinite(b) and np.isfinite(ft) else float("nan")
        deltas.append(d)
        sign = "+" if d > 0 else ""
        print(f"{f:<25s} {b:10.4f} {ft:10.4f} {sign}{d:9.4f}")

    base_dices = [r["dice"] for r in baseline if r["has_gt"]]
    fine_dices = [r["dice"] for r in finetuned if r["has_gt"]]
    valid_deltas = [d for d in deltas if np.isfinite(d)]
    print("-" * 58)
    if base_dices and fine_dices:
        bm, fm = np.mean(base_dices), np.mean(fine_dices)
        sign = "+" if fm - bm > 0 else ""
        print(f"{'MEAN':<25s} {bm:10.4f} {fm:10.4f} {sign}{fm - bm:9.4f}")
        print(f"{'STD':<25s} {np.std(base_dices):10.4f} {np.std(fine_dices):10.4f}")
    if valid_deltas:
        improved = sum(1 for d in valid_deltas if d > 0.001)
        degraded = sum(1 for d in valid_deltas if d < -0.001)
        print(f"\nImproved: {improved}/{len(valid_deltas)}  "
              f"Degraded: {degraded}/{len(valid_deltas)}  "
              f"Unchanged: {len(valid_deltas) - improved - degraded}/{len(valid_deltas)}")


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune on real data and compare Dice scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--data", required=True,
                    help="directory of .npz slices (from convert_mat.py)")
    ap.add_argument("--checkpoint", default=None,
                    help="pretrained checkpoint (.pt) to start from")
    ap.add_argument("--model-config", default=None,
                    help='build model from config instead of a .pt checkpoint, '
                         'e.g. \'{"name":"cinema_ft","dataset":"acdc"}\'')
    ap.add_argument("--normalization-cinema", default="percentile",
                    choices=["zscore", "minmax", "percentile"],
                    help="normalization for CineMA models (default: percentile)")
    ap.add_argument("--split", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="train/val/test fractions (default: 0.6 0.2 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for split")
    ap.add_argument("--out", default="runs/finetune_eval",
                    help="output directory for checkpoints and results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--normalization", default="zscore",
                    choices=["zscore", "minmax", "percentile"])

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping: stop after N epochs without val Dice improvement (0=disabled)")

    # Shortcuts
    ap.add_argument("--skip-train", action="store_true",
                    help="skip training, just compare checkpoints")
    ap.add_argument("--finetuned-checkpoint",
                    help="use this instead of training (implies --skip-train)")
    ap.add_argument("--postprocess", action="store_true",
                    help="apply post-processing to predictions")
    args = ap.parse_args()

    if args.finetuned_checkpoint:
        args.skip_train = True

    if not args.checkpoint and not args.model_config:
        ap.error("either --checkpoint or --model-config is required")

    if args.model_config:
        import json as _json
        args._model_cfg = _json.loads(args.model_config)
        if "cinema" in args._model_cfg.get("name", ""):
            args.normalization = args.normalization_cinema
    else:
        args._model_cfg = None

    device = resolve_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Discover and split data ---
    data_dir = Path(args.data)
    all_files = sorted(data_dir.glob("*.npz"))
    all_files = [f for f in all_files if "_t1true" not in f.name and "_masks" not in f.name]
    if not all_files:
        print(f"No .npz slice files found in {data_dir}")
        sys.exit(1)

    train_files, val_files, test_files = split_files(all_files, args.split, args.seed)
    print(f"Data: {len(all_files)} slices in {data_dir}")
    print(f"Split (seed={args.seed}): train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")

    split_dir = out / "splits"
    for name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        symlink_split(files, split_dir / name)

    split_info = {
        "train": [f.name for f in train_files],
        "val": [f.name for f in val_files],
        "test": [f.name for f in test_files],
        "seed": args.seed,
        "ratios": args.split,
    }
    (out / "split.json").write_text(json.dumps(split_info, indent=1))
    print(f"Split manifest saved to {out / 'split.json'}")

    # --- 2. Baseline inference on test set ---
    print(f"\n{'='*58}")
    if args._model_cfg:
        print(f"BASELINE: {args._model_cfg['name']} (from HuggingFace)")
        model_base = build_model(args._model_cfg).to(device).eval()
        model_cfg = args._model_cfg
        ck = {"model_cfg": model_cfg}
    else:
        print(f"BASELINE: {args.checkpoint}")
        model_base, ck = load_checkpoint(args.checkpoint, device)
        model_cfg = ck.get("model_cfg", {"name": "unet", "in_channels": 1, "n_classes": 2, "base": 32, "depth": 4})
    print(f"{'='*58}")
    n_params = sum(p.numel() for p in model_base.parameters()) / 1e6
    print(f"Model: {model_cfg.get('name', '?')} ({n_params:.2f}M params)")
    if "val_dice" in ck:
        print(f"Checkpoint val DSC: {ck['val_dice']:.3f}")

    baseline_results = run_inference(model_base, test_files, device, args.size, args.normalization)
    base_dices = [r["dice"] for r in baseline_results if r["has_gt"]]
    if base_dices:
        print(f"Baseline test Dice: {np.mean(base_dices):.4f} +/- {np.std(base_dices):.4f} (n={len(base_dices)})")
    else:
        print("WARNING: no ground truth masks found — Dice cannot be computed")

    (out / "baseline_results.json").write_text(json.dumps(baseline_results, indent=1))

    # --- 3. Fine-tune ---
    if not args.skip_train:
        print(f"\n{'='*58}")
        print(f"FINE-TUNING: {args.epochs} epochs, lr={args.lr}")
        print(f"{'='*58}")

        ft_cfg = {
            "device": args.device,
            "plugins": [],
            "data": {
                "train": {
                    "name": "npz_slices",
                    "root": str(split_dir / "train"),
                    "size": args.size,
                    "augment": True,
                    "normalization": args.normalization,
                },
                "val": {
                    "name": "npz_slices",
                    "root": str(split_dir / "val"),
                    "size": args.size,
                    "normalization": args.normalization,
                },
            },
            "model": model_cfg,
            "loss": [
                {"name": "ce", "weight": 1.0},
                {"name": "dice", "weight": 0.5},
            ],
            "optimizer": {"name": "Adam", "lr": args.lr, "weight_decay": args.weight_decay},
            "scheduler": {"name": "CosineAnnealingLR", "T_max": args.epochs},
            "train": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_workers": 0,
                "seed": args.seed,
                "init_from": args.checkpoint,  # None for --model-config (weights already loaded)
                "out_dir": str(out / "checkpoints"),
                "patience": args.patience,
            },
        }
        (out / "finetune_config.json").write_text(json.dumps(ft_cfg, indent=1))
        best_pt, model_ft = train(ft_cfg, model=model_base)
        ft_checkpoint = str(best_pt)
    else:
        ft_checkpoint = args.finetuned_checkpoint or str(out / "checkpoints" / "best.pt")
        print(f"\nSkipping training — using checkpoint: {ft_checkpoint}")
    del model_base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- 4. Fine-tuned inference on the same test set ---
    print(f"\n{'='*58}")
    print(f"FINE-TUNED: {ft_checkpoint}")
    print(f"{'='*58}")
    if not args.skip_train:
        # model_ft already holds the best weights from train()
        ck_ft = torch.load(ft_checkpoint, map_location=device, weights_only=False)
        model_ft.load_state_dict(ck_ft["model"])
        model_ft.eval()
    else:
        model_ft, ck_ft = load_checkpoint(ft_checkpoint, device)
    if "val_dice" in ck_ft:
        print(f"Checkpoint val DSC: {ck_ft['val_dice']:.3f}")

    finetuned_results = run_inference(model_ft, test_files, device, args.size, args.normalization)
    fine_dices = [r["dice"] for r in finetuned_results if r["has_gt"]]
    if fine_dices:
        print(f"Finetuned test Dice: {np.mean(fine_dices):.4f} +/- {np.std(fine_dices):.4f} (n={len(fine_dices)})")

    (out / "finetuned_results.json").write_text(json.dumps(finetuned_results, indent=1))

    # --- 5. Comparison ---
    print_comparison(baseline_results, finetuned_results)

    summary = {
        "baseline_checkpoint": args.checkpoint or json.dumps(args._model_cfg),
        "finetuned_checkpoint": ft_checkpoint,
        "split": split_info,
        "baseline_dice_mean": float(np.mean(base_dices)) if base_dices else None,
        "baseline_dice_std": float(np.std(base_dices)) if base_dices else None,
        "finetuned_dice_mean": float(np.mean(fine_dices)) if fine_dices else None,
        "finetuned_dice_std": float(np.std(fine_dices)) if fine_dices else None,
        "n_test": len(test_files),
        "epochs": args.epochs if not args.skip_train else None,
        "lr": args.lr if not args.skip_train else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nAll results saved to {out}/")


if __name__ == "__main__":
    main()
