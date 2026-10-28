"""Run the full pipeline (segment -> post-process -> align -> fit -> stats) on slice .npz files.

    python scripts/run_pipeline.py -c configs/pipeline/paper_stone.yaml --inputs data/synthetic/test/slice_*[0-9].npz
    python scripts/run_pipeline.py ... --set pipeline.segmenter.name=ground_truth     # bypass the FCN
    python scripts/run_pipeline.py ... --set pipeline.aligner.name=none               # ablate alignment
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


def figure(res, path, t1_true=None):
    K = len(res["tis"])
    fig, ax = plt.subplots(3, max(K, 4), figsize=(2 * max(K, 4), 6.5))
    for a in ax.ravel():
        a.axis("off")
    for k in range(K):
        ax[0, k].imshow(res["images"][k], cmap="gray")
        ax[0, k].contour(res["masks"][k], [0.5], colors="r" if res["ok"][k] else "y", linewidths=0.6)
        ax[0, k].set_title(f"TI={res['tis'][k]:.0f}" + ("" if res["ok"][k] else " ✗"), fontsize=8)
    if res["status"] == "ok":
        if res["aligned"].ndim == 3:  # 2-D common grid (e.g. polar): show it
            ax[1, 0].imshow(res["aligned"][0], aspect="auto", cmap="gray")
            ax[1, 0].set_title("aligned T1w (common grid)", fontsize=8)
            ax[1, 1].imshow(res["t1_common"], aspect="auto", cmap="jet", vmin=800, vmax=1400)
            ax[1, 1].set_title("T1 on common grid", fontsize=8)
        ref = res["images"][res["ref_index"]]
        ys, xs = np.nonzero(res["masks"][res["ref_index"]])
        sl = np.s_[max(ys.min() - 15, 0):ys.max() + 15, max(xs.min() - 15, 0):xs.max() + 15]
        ax[2, 0].imshow(ref[sl], cmap="gray")
        im = ax[2, 0].imshow(res["t1_map"][sl], cmap="jet", vmin=800, vmax=1400)
        ax[2, 0].set_title("T1 map (ms)", fontsize=8)
        fig.colorbar(im, ax=ax[2, 0], fraction=0.046)
        ax[2, 1].imshow(ref[sl], cmap="gray")
        ax[2, 1].imshow(np.where(res["midwall"], res["t1_map"], np.nan)[sl], cmap="jet", vmin=800, vmax=1400)
        s = res["stats"]
        ax[2, 1].set_title(f"mid-wall {s['mean']:.0f}±{s['sd']:.0f} ms", fontsize=8)
        if t1_true is not None:
            ax[2, 2].imshow(np.where(res["masks"][res["ref_index"]], t1_true, np.nan)[sl], cmap="jet",
                            vmin=800, vmax=1400)
            ax[2, 2].set_title("ground-truth T1", fontsize=8)
    fig.suptitle(f"{Path(path).stem}: {res['status']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/pipeline/paper_stone.yaml")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/pipeline")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    pipe = Pipeline.from_config(load_config(args.config, args.set))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for f in args.inputs:
        d = load_slice(f)
        res = pipe(d["images"], d["tis"], d["pixel_spacing"], masks=d.get("masks"))
        row = dict(file=f, status=res["status"], n_ok=int(res["ok"].sum()))
        if "masks" in d:
            gt = pad_crop(d["masks"], pipe.size)
            dsc = [dice(res["raw_masks"][k], gt[k]) for k in range(len(gt))]
            row.update(dice_mean=float(np.mean(dsc)), dice_sd=float(np.std(dsc)))
        t1_true = None
        truth = Path(f).with_name(Path(f).stem + "_t1true.npz")
        if truth.exists() and res["status"] == "ok":  # synthetic ground truth, per frame
            t1_true = pad_crop(np.load(truth)["t1"], pipe.size)[res["ref_index"]]
        if res["status"] == "ok":
            row.update(res["stats"])
            if t1_true is not None:
                m = res["midwall"] & np.isfinite(res["t1_map"])
                row["t1_true_midwall"] = float(t1_true[m].mean())
                row["mae_ms"] = float(np.abs(res["t1_map"][m] - t1_true[m]).mean())
        figure(res, out / (Path(f).stem + ".png"), t1_true)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
        summary.append(row)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
