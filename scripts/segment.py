"""Segmentation only: masks + post-processing for slice .npz files (stages from the pipeline config).

    python scripts/segment.py -c configs/pipeline/paper_stone.yaml --inputs data/synthetic/test/slice_*[0-9].npz
    python scripts/segment.py ... --set pipeline.segmenter.checkpoint=runs/smp_resnet34/best.pt pipeline.postprocess=[]
Writes <out>/<name>_masks.npz (raw + post-processed masks), an overlay PNG, and Dice if GT masks exist.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from t1seg.analysis import dice
from t1seg.config import load_config
from t1seg.io import load_slice, pad_crop
from t1seg.pipeline import Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/pipeline/paper_stone.yaml")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/seg")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    pipe = Pipeline.from_config(load_config(args.config, args.set))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in args.inputs:
        d = load_slice(f)
        r = pipe.segment_only(d["images"], d["tis"], d["pixel_spacing"], masks=d.get("masks"))
        imgs, raw, masks, ok = r["images"], r["raw_masks"], r["masks"], r["ok"]
        np.savez_compressed(out / f"{Path(f).stem}_masks.npz", raw=raw, masks=masks, ok=ok)

        row = dict(file=f, n_ok=int(ok.sum()), n=len(ok))
        if "masks" in d:
            gt = pad_crop(d["masks"], pipe.size)
            row["dice_raw"] = float(np.mean([dice(raw[k], gt[k]) for k in range(len(gt))]))
            row["dice_post"] = float(np.mean([dice(masks[k], gt[k]) for k in range(len(gt))]))
        rows.append(row)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))

        fig, ax = plt.subplots(1, len(imgs), figsize=(1.8 * len(imgs), 2.2))
        for k, a in enumerate(np.atleast_1d(ax)):
            a.imshow(imgs[k], cmap="gray")
            if "masks" in d:
                a.contour(gt[k], [0.5], colors="lime", linewidths=0.5)
            a.contour(masks[k], [0.5], colors="r" if ok[k] else "y", linewidths=0.5)
            a.set_title(f"TI={d['tis'][k]:.0f}", fontsize=7)
            a.axis("off")
        fig.suptitle("red = prediction (yellow = failed QC), green = ground truth", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{Path(f).stem}.png", dpi=110)
        plt.close(fig)
    json.dump(rows, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
