#!/bin/bash
#SBATCH --job-name=cinema_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cd /t1seg || exit 1

source .venv/bin/activate

python scripts/finetune_eval.py \
    --data data/real_slices \
    --model-config '{"name":"cinema_ft","dataset":"acdc"}' \
    --epochs 15 \
    --device auto
