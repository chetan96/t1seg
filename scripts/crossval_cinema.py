"""K-fold cross-validation for CineMA (or any model) fine-tuning on real data.

Splits .npz slices into K folds, fine-tunes per fold with early stopping,
and reports aggregate Dice scores across all folds.

Examples
--------
# 5-fold CV with CineMA ACDC, early stopping after 3 stale epochs
python scripts/crossval_cinema.py \
    --data data/real_slices \
    --model-config '{"name":"cinema_ft","dataset":"acdc"}' \
    --epochs 20 --patience 3

# Resume fold 0 from an earlier run's checkpoint
python scripts/crossval_cinema.py \
    --data data/real_slices \
    --model-config '{"name":"cinema_ft","dataset":"acdc"}' \
    --epochs 20 --patience 3 \
    --resume-fold 0 --resume-checkpoint runs/finetune_eval/checkpoints/last.pt
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from t1seg.analysis import dice
from t1seg.data import normalize
from t1seg.device import resolve_device
from t1seg.io import load_slice, pad_crop
from t1seg.models import build_model, load_checkpoint
from t1seg.train import train, evaluate


def kfold_split(files, k, seed, val_ratio=0.25):
    """Split files into K folds. Returns list of (train, val, test) tuples."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    fold_size = len(files) // k
    folds = []
    for i in range(k):
        test_idx = idx[i * fold_size:(i + 1) * fold_size]
        train_val_idx = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
        n_val = max(1, int(len(train_val_idx) * val_ratio))
        val_idx = train_val_idx[:n_val]
        train_idx = train_val_idx[n_val:]
        folds.append((
            [files[j] for j in train_idx],
            [files[j] for j in val_idx],
            [files[j] for j in test_idx],
        ))
    return folds


def symlink_split(files, dest):
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in files:
        (dest / f.name).symlink_to(f.resolve())


@torch.no_grad()
def run_inference(model, files, device, size=256, normalization="zscore"):
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


def run_fold(fold_idx, train_files, val_files, test_files, args, device, out):
    """Run one fold: baseline → fine-tune → evaluate → save results."""
    fold_dir = out / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    split_dir = fold_dir / "splits"
    for name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        symlink_split(files, split_dir / name)

    split_info = {
        "train": [f.name for f in train_files],
        "val": [f.name for f in val_files],
        "test": [f.name for f in test_files],
    }
    (fold_dir / "split.json").write_text(json.dumps(split_info, indent=1))

    model_cfg = args._model_cfg
    normalization = args.normalization

    # --- Baseline ---
    print(f"  Building model: {model_cfg['name']}...")
    model = build_model(model_cfg).to(device).eval()
    baseline_results = run_inference(model, test_files, device, args.size, normalization)
    base_dices = [r["dice"] for r in baseline_results if r["has_gt"]]
    if base_dices:
        print(f"  Baseline test Dice: {np.mean(base_dices):.4f} ± {np.std(base_dices):.4f}")
    (fold_dir / "baseline_results.json").write_text(json.dumps(baseline_results, indent=1))

    # --- Fine-tune ---
    ck_dir = str(fold_dir / "checkpoints")
    ft_cfg = {
        "device": args.device,
        "plugins": [],
        "data": {
            "train": {"name": "npz_slices", "root": str(split_dir / "train"),
                      "size": args.size, "augment": True, "normalization": normalization},
            "val": {"name": "npz_slices", "root": str(split_dir / "val"),
                    "size": args.size, "normalization": normalization},
        },
        "model": model_cfg,
        "loss": [{"name": "ce", "weight": 1.0}, {"name": "dice", "weight": 0.5}],
        "optimizer": {"name": "Adam", "lr": args.lr, "weight_decay": args.weight_decay},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": args.epochs},
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": 0,
            "seed": args.seed,
            "out_dir": ck_dir,
            "patience": args.patience,
        },
    }

    if args.resume_fold == fold_idx and args.resume_checkpoint:
        ft_cfg["train"]["resume_from"] = args.resume_checkpoint
        print(f"  Resuming from {args.resume_checkpoint}")

    (fold_dir / "finetune_config.json").write_text(json.dumps(ft_cfg, indent=1))
    best_pt, model = train(ft_cfg, model=model)

    # Load best weights (train() returns last-epoch model)
    ck_ft = torch.load(str(best_pt), map_location=device, weights_only=False)
    model.load_state_dict(ck_ft["model"])
    model.eval()

    if "val_dice" in ck_ft:
        print(f"  Best val DSC: {ck_ft['val_dice']:.3f}")

    # --- Evaluate fine-tuned ---
    finetuned_results = run_inference(model, test_files, device, args.size, normalization)
    fine_dices = [r["dice"] for r in finetuned_results if r["has_gt"]]
    if fine_dices:
        print(f"  Finetuned test Dice: {np.mean(fine_dices):.4f} ± {np.std(fine_dices):.4f}")
    (fold_dir / "finetuned_results.json").write_text(json.dumps(finetuned_results, indent=1))

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "fold": fold_idx,
        "n_train": len(train_files),
        "n_val": len(val_files),
        "n_test": len(test_files),
        "baseline_dice_mean": float(np.mean(base_dices)) if base_dices else None,
        "baseline_dice_std": float(np.std(base_dices)) if base_dices else None,
        "finetuned_dice_mean": float(np.mean(fine_dices)) if fine_dices else None,
        "finetuned_dice_std": float(np.std(fine_dices)) if fine_dices else None,
        "best_checkpoint": str(best_pt),
        "stopped_epoch": ck_ft.get("epoch"),
    }


def main():
    ap = argparse.ArgumentParser(
        description="K-fold cross-validation for fine-tuning on real data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--data", required=True, help="directory of .npz slices")
    ap.add_argument("--model-config", required=True,
                    help='model config JSON, e.g. \'{"name":"cinema_ft","dataset":"acdc"}\'')
    ap.add_argument("--folds", type=int, default=5, help="number of folds (default: 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/crossval", help="output directory")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--normalization", default="zscore",
                    choices=["zscore", "minmax", "percentile"])
    ap.add_argument("--normalization-cinema", default="percentile",
                    choices=["zscore", "minmax", "percentile"])

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3,
                    help="early stopping patience on val Dice (0=disabled)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--resume-fold", type=int, default=None,
                    help="fold index to resume training from a checkpoint")
    ap.add_argument("--resume-checkpoint", default=None,
                    help="checkpoint path for --resume-fold")
    args = ap.parse_args()

    args._model_cfg = json.loads(args.model_config)
    if "cinema" in args._model_cfg.get("name", ""):
        args.normalization = args.normalization_cinema

    device = resolve_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data)
    all_files = sorted(data_dir.glob("*.npz"))
    all_files = [f for f in all_files if "_t1true" not in f.name and "_masks" not in f.name]
    if not all_files:
        print(f"No .npz slice files found in {data_dir}")
        sys.exit(1)

    folds = kfold_split(all_files, args.folds, args.seed)

    print(f"{'='*60}")
    print(f"{args.folds}-FOLD CROSS-VALIDATION")
    print(f"Data: {len(all_files)} slices in {data_dir}")
    print(f"Model: {args._model_cfg['name']}")
    print(f"Max epochs: {args.epochs}, patience: {args.patience}")
    print(f"{'='*60}\n")

    fold_results = []
    for i, (train_f, val_f, test_f) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {i}/{args.folds-1}  (train={len(train_f)} val={len(val_f)} test={len(test_f)})")
        print(f"  Test files: {[f.name for f in test_f]}")
        print(f"{'='*60}")

        result = run_fold(i, train_f, val_f, test_f, args, device, out)
        fold_results.append(result)

        (out / "results_so_far.json").write_text(json.dumps(fold_results, indent=1))

    # --- Aggregate ---
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({args.folds}-fold CV)")
    print(f"{'='*60}")

    base_means = [r["baseline_dice_mean"] for r in fold_results if r["baseline_dice_mean"] is not None]
    fine_means = [r["finetuned_dice_mean"] for r in fold_results if r["finetuned_dice_mean"] is not None]

    print(f"\n{'Fold':<8s} {'Baseline':>10s} {'Finetuned':>10s} {'Delta':>10s} {'Stopped':>8s}")
    print("-" * 50)
    for r in fold_results:
        b = r["baseline_dice_mean"] or float("nan")
        f = r["finetuned_dice_mean"] or float("nan")
        d = f - b if np.isfinite(b) and np.isfinite(f) else float("nan")
        sign = "+" if d > 0 else ""
        ep = r.get("stopped_epoch", "?")
        print(f"Fold {r['fold']:<4d} {b:10.4f} {f:10.4f} {sign}{d:9.4f} {ep:>8}")
    print("-" * 50)

    if base_means and fine_means:
        bm, bs = np.mean(base_means), np.std(base_means)
        fm, fs = np.mean(fine_means), np.std(fine_means)
        sign = "+" if fm - bm > 0 else ""
        print(f"{'MEAN':<8s} {bm:10.4f} {fm:10.4f} {sign}{fm-bm:9.4f}")
        print(f"{'STD':<8s} {bs:10.4f} {fs:10.4f}")

    summary = {
        "model": args._model_cfg,
        "folds": args.folds,
        "seed": args.seed,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "n_slices": len(all_files),
        "baseline_dice_mean": float(np.mean(base_means)) if base_means else None,
        "baseline_dice_std": float(np.std(base_means)) if base_means else None,
        "finetuned_dice_mean": float(np.mean(fine_means)) if fine_means else None,
        "finetuned_dice_std": float(np.std(fine_means)) if fine_means else None,
        "per_fold": fold_results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nAll results saved to {out}/")


if __name__ == "__main__":
    main()
