# t1seg — automated myocardial T1 mapping analysis

PyTorch re-implementation of the pipeline in *Fahmy et al., "Automated analysis of CMR myocardial
native T1 mapping images using fully convolutional neural networks", JCMR 2019*
([doi:10.1186/s12968-018-0516-1](https://doi.org/10.1186/s12968-018-0516-1)).
The authors did not release code, so this is built from the Methods section.

**Status:** pipeline complete and unit-tested on synthetic data; real-data converter and
CineMA transfer baselines ready. An 8-epoch synthetic run (~6 min on an M2 via MPS) reaches
val DSC ≈ 0.60 — train longer (`epochs: 30+`) for meaningful masks. Paper reports
DSC 0.85 ± 0.07 on real STONE data. See [USAGE.md](USAGE.md) for the full walkthrough.

## Design: every stage is a plug-in

```
images ─▶ segmenter ─▶ postprocess steps ─▶ aligner ─▶ fitter ─▶ analysis
          (torch ckpt   (largest_component,  (polar,     (2p, 3p,
           any model,    shape_qc, refine,    none, …)    …)
           GT, …)        …)
training: dataset ─▶ model ─▶ loss terms ─▶ any torch.optim / lr_scheduler
```

Each stage is a **registry** (`t1seg/registry.py`) and configs pick components **by name + kwargs**.
New paper / new idea → write one class, decorate it, reference it in YAML. No existing code changes.

| Registry | Interface | Built-ins |
|---|---|---|
| `MODELS` | `nn.Module`: `(B,C,H,W) → logits (B,n_classes,H,W)` | `unet` (paper), `smp` (100+ encoders), `monai` (SwinUNETR, AttentionUnet…), `torchvision_deeplab` |
| `LOSSES` | `f(logits, target) → scalar`, weighted list in YAML | `ce`, `dice`, `focal` |
| `DATASETS` | torch `Dataset` → `(image, mask)` | `npz_slices` |
| `SEGMENTERS` | `f(images (K,H,W), tis) → masks` | `torch` (any checkpoint), `ground_truth` |
| `POSTPROCESS` | factory → `f(masks, ok, tis, spacing) → (masks, ok)` | `largest_component`, `shape_qc`, `refine_from_reference` |
| `ALIGNERS` | `.align(images, masks, ok, tis) → Aligned` | `polar` (paper), `none` |
| `FITTERS` | `f(stack (K,…), tis) → T1 (…)` | `2p` (STONE), `3p` (MOLLI) |

`python -m t1seg [plugin.py …]` lists everything registered plus available devices.

**Hardware:** `device: auto` (default) picks **CUDA → Apple MPS → CPU**. An explicit `device: cuda` / `cuda:1` /
`mps` / `cpu` is honoured, and an unavailable one is an error rather than a silent CPU fallback.

**Adding a method** — see `examples/plugin_example.py` (new model, post-processing step and fitter):

```python
from t1seg.registry import MODELS

@MODELS.register("my_net")
class MyNet(nn.Module): ...
```
```yaml
plugins: [examples/plugin_example.py]     # or a dotted module name
model: {name: my_net, width: 32}
```

Checkpoints store the model config + plugin list, so any architecture reloads with `load_checkpoint()`.

**Override anything from the CLI** (`--set key.path=value`, YAML-parsed). Changing `X.name` to another
component resets that block; then add its kwargs:

```bash
python scripts/train.py -c configs/train/unet_paper.yaml --set train.epochs=50 model.dropout=0.5 device=cuda
python scripts/train.py -c configs/train/unet_paper.yaml --set model.name=smp model.encoder_name=resnet50 train.out_dir=runs/r50
python scripts/run_pipeline.py --inputs … --set pipeline.aligner.name=none          # ablation
```

## Setup

```bash
conda create -n t1seg python=3.11 && conda activate t1seg
pip install -e .              # core
pip install -e ".[zoo]"       # + segmentation_models_pytorch, monai model zoos
```

Uses CUDA if available, else Apple Silicon MPS, else CPU. On a Mac, make sure your conda is a
native arm64 install (`python -c "import platform; print(platform.machine())"` → `arm64`);
an x86/Rosetta conda gets no GPU and an old PyTorch.

## Quick start (synthetic data)

```bash
python scripts/make_synthetic.py --n-train 60 --n-val 10 --n-test 5
python scripts/train.py   -c configs/train/unet_paper.yaml
python scripts/segment.py -c configs/pipeline/paper_stone.yaml --inputs data/synthetic/test/slice_*[0-9].npz
python scripts/run_pipeline.py -c configs/pipeline/paper_stone.yaml --inputs data/synthetic/test/slice_*[0-9].npz
pytest -q tests
```

Figures + `summary.json` land in `outputs/`. Each training run writes `config.json`, `log.json`, `best.pt`, `last.pt`.

Synthetic sanity check (GT masks, 3 slices): T1 error ≈ 10 ms with `polar` alignment vs 38–112 ms with `none`.

## Training workflows

Four workflows are supported end to end. Each has a dedicated config and can be
customised with `--set` overrides. See [USAGE.md](USAGE.md) for the full details.

### 1. Train from scratch

Train on synthetic data with randomly initialised weights:

```bash
# Paper U-Net (~7.8M params)
python scripts/train.py -c configs/train/unet_paper.yaml

# SMP ResNet-34 U-Net (no pretrained encoder)
python scripts/train.py -c configs/train/smp_resnet34.yaml
```

### 2. Pretrain (ImageNet encoder)

SMP U-Net with an ImageNet-pretrained ResNet-34 encoder — the encoder starts with
strong low-level features, only the decoder trains from scratch:

```bash
python scripts/train.py -c configs/train/smp_pretrained.yaml

# Other encoders / architectures
python scripts/train.py -c configs/train/smp_pretrained.yaml \
    --set model.encoder_name=resnet50
python scripts/train.py -c configs/train/smp_pretrained.yaml \
    --set model.arch=UnetPlusPlus model.encoder_name=efficientnet-b3
```

Requires: `pip install segmentation-models-pytorch`

### 3. Fine-tune on real data

Take a checkpoint pretrained on synthetic data and fine-tune it on real T1
mapping slices (lower LR, fewer epochs):

```bash
# Convert real .mat data to .npz first
python scripts/convert_mat.py \
    --images /path/to/images.mat --masks /path/to/masks.mat \
    --out data/real_slices --filter-empty

# Fine-tune (default: loads runs/unet_paper/best.pt)
python scripts/train.py -c configs/train/finetune_real.yaml

# Fine-tune from an SMP pretrained checkpoint instead
python scripts/train.py -c configs/train/finetune_real.yaml \
    --set train.init_from=runs/smp_pretrained/best.pt \
           model.name=smp model.encoder_name=resnet34 model.encoder_weights=null
```

### 4. Inference

**Standalone** — only needs a checkpoint and `.npz` input files:

```bash
# Model predictions only
python scripts/inference.py --checkpoint runs/unet_paper/best.pt \
    --inputs data/synthetic/test/*.npz --out outputs/inference

# With paper post-processing
python scripts/inference.py --checkpoint runs/unet_paper/best.pt \
    --inputs data/synthetic/test/*.npz --out outputs/inference --postprocess

# Full pipeline (segment -> align -> T1 fit -> stats)
python scripts/inference.py --checkpoint runs/unet_paper/best.pt \
    --inputs data/synthetic/test/*.npz --out outputs/full --full-pipeline
```

**Pipeline-config-driven** — full control over each stage:

```bash
# Segmentation only (masks + Dice)
python scripts/segment.py -c configs/pipeline/paper_stone.yaml \
    --inputs data/synthetic/test/*.npz

# Full pipeline with T1 map
python scripts/run_pipeline.py -c configs/pipeline/paper_stone.yaml \
    --inputs data/synthetic/test/*.npz

# CineMA transfer baseline on real data
python scripts/segment.py -c configs/pipeline/cinema_acdc.yaml \
    --inputs data/real_slices/*.npz --out outputs/cinema_acdc
```

Output per slice: `_masks.npz` (predictions), `.png` (overlay), `summary.json` (Dice + stats).

## Using real data

The `npz_slices` dataset reads one `.npz` per slice: `images (K,H,W)`, `tis (K,)` ms (`np.inf` allowed),
optional `masks (K,H,W)`, `pixel_spacing (2,)`. Use `scripts/convert_mat.py` to convert `.mat` files,
`t1seg/io.py:load_dicom_series` for DICOMs, or register your own `DATASETS` / `SEGMENTERS` entry.

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
| cross-entropy | `loss: [ce, {name: dice, weight: 0.5}]` (drop the dice term for paper) | thin annulus |
| transfer learning from an LGE model | `init_from:` hook, no weights | not public |
| STONE, 2-param fit | 2p and 3p (MOLLI Look-Locker + polarity restoration) | MOLLI is what most sites acquire |
| eccentricity ≤ 0.65 | same constant | re-derive from your training masks |
| per-slice "regional" T1 | per-slice only | AHA segments need RV insertion points — next step |

Good next steps: real data loader, AHA 16-segment analysis (RV insertion detection), uncertainty / QC,
nnU-Net baseline for comparison, Bland–Altman vs manual ROI.
