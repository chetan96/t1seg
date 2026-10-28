# t1seg — automated myocardial T1 mapping analysis

PyTorch re-implementation of the pipeline in *Fahmy et al., "Automated analysis of CMR myocardial
native T1 mapping images using fully convolutional neural networks", JCMR 2019*
([doi:10.1186/s12968-018-0516-1](https://doi.org/10.1186/s12968-018-0516-1)).
The authors did not release code, so this is built from the Methods section.

**Status:** pipeline complete and unit-tested on synthetic data; no real data wired in yet.
An 8-epoch synthetic run (~6 min on an M2 via MPS) reaches val DSC ≈ 0.60 — train longer
(`epochs: 30+`) for meaningful masks. Paper reports DSC 0.85 ± 0.07 on real STONE data.

```
T1w images (one slice, K inversion times)
  └─ U-Net segmentation of LV myocardium, per image            t1seg/model.py
      └─ largest object, >5 cm², Euler==0, eccentricity<=0.65   t1seg/postprocess.py
          └─ failed masks: affine-fit the longest-TI mask onto them
              └─ polar resampling per image (360 rays × C=20)    t1seg/polar.py   ← implicit motion correction
                  └─ pixel-wise fit (2-param STONE / 3-param MOLLI) t1seg/t1fit.py
                      └─ inverse polar onto shortest-TI image grid
                          └─ mid-wall band (⅓ wall thickness), 850–1500 ms range   t1seg/analysis.py
```

## Setup

```bash
conda create -n t1seg python=3.11 && conda activate t1seg
pip install -e .          # torch, numpy, scipy, scikit-image, ...
```

Uses CUDA if available, else Apple Silicon MPS, else CPU. On a Mac, make sure your conda is a
native arm64 install (`python -c "import platform; print(platform.machine())"` → `arm64`);
an x86/Rosetta conda gets no GPU and an old PyTorch.

## Quick start (synthetic data)

```bash
python scripts/make_synthetic.py --n-train 60 --n-val 10 --n-test 5
python scripts/train.py -c configs/default.yaml --epochs 8
python scripts/segment.py      --ckpt runs/default/best.pt --inputs data/synthetic/test/slice_*[0-9].npz   # segmentation only
python scripts/run_pipeline.py --ckpt runs/default/best.pt --inputs data/synthetic/test/slice_*[0-9].npz   # full T1 pipeline
python scripts/run_pipeline.py --use-gt-masks --inputs ...     # skip the FCN, test the rest
pytest -q tests
```

Figures + `summary.json` land in `outputs/`.

## Using real data

Everything reads one `.npz` per slice: `images (K,H,W)`, `tis (K,)` ms (`np.inf` allowed),
optional `masks (K,H,W)`, `pixel_spacing (2,)`. Write a converter to that format
(`t1seg/io.py` has `save_slice` and a best-effort `load_dicom_series`) and point the config at it.

Public data options:
- **CMRxRecon (2023/2024)** — multi-coil k-space plus reconstructed T1 mapping (MOLLI, 9 TIs) for ~300
  subjects; the release notes mention segmentation labels for some subsets — verify that T1-map
  myocardium labels are included before relying on it. Register on Synapse.
  Use `--fit-model 3p`.
- **Your own scanner data**: annotate a few slices in ITK-SNAP / 3D Slicer → `.npz`.

## Differences from the paper / things to revisit

| Paper | Here | Why |
|---|---|---|
| TensorFlow, ~9M params, 149 layers | PyTorch U-Net, ~7.8M params | exact layer layout isn't given |
| dropout 0.5 in every bottleneck | 0.2 (configurable) | trains faster from scratch |
| cross-entropy | CE + 0.5·soft Dice (set `dice_weight: 0` for paper) | thin annulus |
| transfer learning from an LGE model | `init_from:` hook, no weights | not public |
| STONE, 2-param fit | 2p and 3p (MOLLI Look-Locker + polarity restoration) | MOLLI is what most sites acquire |
| eccentricity ≤ 0.65 | same constant | re-derive from your training masks |
| per-slice "regional" T1 | per-slice only | AHA segments need RV insertion points — next step |

Good next steps: real data loader, AHA 16-segment analysis (RV insertion detection), uncertainty / QC,
nnU-Net baseline for comparison, Bland–Altman vs manual ROI.
