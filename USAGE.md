# T1 Segmentation Pipeline — Usage Guide

Complete walkthrough for training, fine-tuning, and running inference with the
Fahmy et al. (JCMR 2019) T1 mapping segmentation pipeline.

All commands assume you're in the project root and using the `t1seg` conda env:

```bash
cd ~/Desktop/MRI/T1_segment_research
conda activate t1seg
```

---

## 0. Generate Synthetic Data

Synthetic data is needed for training from scratch or pretraining.
Skip this if `data/synthetic/` already has files.

```bash
# Default: 60 train / 15 val / 5 test slices (STONE TI scheme)
python scripts/make_synthetic.py

# Larger dataset for better training
python scripts/make_synthetic.py --n-train 200 --n-val 40 --n-test 20

# MOLLI TI scheme instead of STONE
python scripts/make_synthetic.py --scheme molli --out data/synthetic_molli
```

Output: `data/synthetic/{train,val,test}/slice_NNNN.npz`

---

## 1. Train from Scratch

Train a U-Net on synthetic data with no pretrained weights.

### Paper U-Net (custom architecture, ~7.8M params)

```bash
python scripts/train.py -c configs/train/unet_paper.yaml
```

### SMP ResNet-34 U-Net (no ImageNet weights)

```bash
python scripts/train.py -c configs/train/smp_resnet34.yaml
```

### Common overrides

```bash
# Fewer epochs for a quick test
python scripts/train.py -c configs/train/unet_paper.yaml --set train.epochs=5

# Change learning rate
python scripts/train.py -c configs/train/unet_paper.yaml --set optimizer.lr=5e-4

# Force CPU
python scripts/train.py -c configs/train/unet_paper.yaml --set device=cpu

# Change output directory
python scripts/train.py -c configs/train/unet_paper.yaml --set train.out_dir=runs/my_experiment
```

Output: `runs/unet_paper/best.pt`, `runs/unet_paper/last.pt`, `runs/unet_paper/log.json`

---

## 2. Pretrain (ImageNet Encoder)

Use a ResNet-34 encoder pretrained on ImageNet. The encoder starts with strong
low-level features; only the decoder trains from scratch. This typically
converges faster and to a higher Dice than training from scratch.

```bash
python scripts/train.py -c configs/train/smp_pretrained.yaml
```

Requires: `pip install segmentation-models-pytorch`

### Other pretrained encoders

```bash
# ResNet-50 (larger, more capacity)
python scripts/train.py -c configs/train/smp_pretrained.yaml \
    --set model.encoder_name=resnet50

# EfficientNet-B3 (good accuracy/speed tradeoff)
python scripts/train.py -c configs/train/smp_pretrained.yaml \
    --set model.encoder_name=efficientnet-b3

# Use UNet++ instead of plain UNet
python scripts/train.py -c configs/train/smp_pretrained.yaml \
    --set model.arch=UnetPlusPlus
```

Output: `runs/smp_pretrained/best.pt`

---

## 3. Fine-tune on Real Data

Take a checkpoint pretrained on synthetic data (or ImageNet) and fine-tune it
on real T1 mapping slices. Uses lower learning rate and fewer epochs.

### Step 3a: Convert real .mat data to .npz

```bash
# Inspect the .mat files first
python scripts/convert_mat.py \
    --images /path/to/moco_raw_imgs_t1_resize.mat \
    --masks /path/to/endo_epi_masks.mat --list

# Convert slices that have ground-truth masks
python scripts/convert_mat.py \
    --images /path/to/moco_raw_imgs_t1_resize.mat \
    --masks /path/to/endo_epi_masks.mat \
    --out data/real_slices --filter-empty
```

### Step 3b: Split into train/val

```bash
mkdir -p data/real_slices/train data/real_slices/val
# Move ~80% to train, ~20% to val (adjust as needed)
# Example: if you have slices 0-9, put 0-7 in train and 8-9 in val
mv data/real_slices/slice_000{0..7}.npz data/real_slices/train/
mv data/real_slices/slice_000{8,9}.npz data/real_slices/val/
```

### Step 3c: Fine-tune

```bash
# Fine-tune the paper U-Net (pretrained on synthetic)
python scripts/train.py -c configs/train/finetune_real.yaml

# Fine-tune the SMP pretrained model instead
python scripts/train.py -c configs/train/finetune_real.yaml \
    --set train.init_from=runs/smp_pretrained/best.pt \
           model.name=smp model.encoder_name=resnet34 model.encoder_weights=null

# If you have very few real slices, use more aggressive augmentation
python scripts/train.py -c configs/train/finetune_real.yaml \
    --set train.epochs=30 optimizer.lr=5e-5
```

Output: `runs/finetune_real/best.pt`

---

## 4. Inference

### 4a. Simple inference (checkpoint + images -> masks)

The standalone inference script needs only a checkpoint and .npz input files:

```bash
# Basic — model predictions only
python scripts/inference.py \
    --checkpoint runs/unet_paper/best.pt \
    --inputs data/synthetic/test/*.npz \
    --out outputs/inference

# With paper post-processing (largest component + shape QC + reference refinement)
python scripts/inference.py \
    --checkpoint runs/unet_paper/best.pt \
    --inputs data/synthetic/test/*.npz \
    --out outputs/inference --postprocess

# Full pipeline (segment -> polar align -> T1 fit -> midwall stats)
python scripts/inference.py \
    --checkpoint runs/unet_paper/best.pt \
    --inputs data/synthetic/test/*.npz \
    --out outputs/full --full-pipeline

# On real data with the fine-tuned model
python scripts/inference.py \
    --checkpoint runs/finetune_real/best.pt \
    --inputs data/real_slices/*.npz \
    --out outputs/real_inference --postprocess
```

Output per slice:
- `<name>_masks.npz` — predicted masks + ok flags
- `<name>.png` — overlay visualisation (red=pred, green=GT)
- `<name>_t1map.npz` — T1 map + midwall mask (with `--full-pipeline`)
- `summary.json` — Dice scores and stats for all slices

### 4b. Pipeline-config-driven inference

For more control over each pipeline stage (aligner, fitter, post-processing):

```bash
# Segmentation only (masks + Dice)
python scripts/segment.py \
    -c configs/pipeline/paper_stone.yaml \
    --inputs data/synthetic/test/*.npz \
    --out outputs/seg

# Full pipeline (segment -> align -> fit -> T1 map + stats)
python scripts/run_pipeline.py \
    -c configs/pipeline/paper_stone.yaml \
    --inputs data/synthetic/test/*.npz \
    --out outputs/pipeline

# Use ground-truth masks (isolate fitting from segmentation errors)
python scripts/run_pipeline.py \
    -c configs/pipeline/paper_stone.yaml \
    --inputs data/synthetic/test/*.npz \
    --set pipeline.segmenter.name=ground_truth

# CineMA transfer baseline
python scripts/segment.py \
    -c configs/pipeline/cinema_acdc.yaml \
    --inputs data/real_slices/*.npz \
    --out outputs/cinema_acdc

# Disable alignment (ablation)
python scripts/run_pipeline.py \
    -c configs/pipeline/paper_stone.yaml \
    --inputs data/synthetic/test/*.npz \
    --set pipeline.aligner.name=none

# Skip post-processing
python scripts/segment.py \
    -c configs/pipeline/paper_stone.yaml \
    --inputs data/synthetic/test/*.npz \
    --set pipeline.postprocess=[]
```

---

## Quick Reference: Recommended Workflow

```
1.  make_synthetic.py          Generate synthetic training data
          |
2a. train.py (unet_paper)     Train from scratch on synthetic
2b. train.py (smp_pretrained)  OR pretrain with ImageNet encoder
          |
3.  convert_mat.py             Convert real .mat -> .npz
          |
4.  train.py (finetune_real)   Fine-tune on real data
          |
5.  inference.py               Run inference + evaluate
    run_pipeline.py            Full pipeline with T1 fitting
```

## Config Override Syntax

Any config value can be overridden from the command line with `--set`:

```bash
--set key=value                    # top-level
--set train.epochs=50              # nested
--set model.name=smp               # changing component resets its kwargs
--set pipeline.postprocess=[]      # empty list = disable
--set pipeline.segmenter.name=ground_truth  # swap segmenter
```

## Available Components

| Stage        | Registry names                              |
|-------------|---------------------------------------------|
| Model       | `unet`, `smp`, `monai`, `torchvision_deeplab` |
| Segmenter   | `torch`, `ground_truth`, `cinema`           |
| Post-process| `largest_component`, `shape_qc`, `refine_from_reference` |
| Aligner     | `polar`, `none`                             |
| Fitter      | `2p` (STONE), `3p` (MOLLI)                 |
| Loss        | `ce`, `dice`, `focal`                       |
| Dataset     | `npz_slices`                                |
