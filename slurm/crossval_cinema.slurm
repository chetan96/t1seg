#!/bin/bash
#SBATCH --job-name=cinema_cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cd /t1seg || exit 1

source .venv/bin/activate

python scripts/crossval_cinema.py \
    --data data/real_slices \
    --model-config '{"name":"cinema_ft","dataset":"acdc"}' \
    --epochs 20 --patience 3 --folds 5 --seed 42 \
    --resume-fold 4 --resume-checkpoint runs/finetune_eval/checkpoints/last.pt \
    --out runs/crossval_cinema_acdc
