"""Train a segmentation model from a config.

    python scripts/train.py -c configs/train/unet_paper.yaml
    python scripts/train.py -c configs/train/unet_paper.yaml --set train.epochs=5 device=cpu model.base=16
"""
import argparse

from t1seg.config import load_config
from t1seg.train import train

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="configs/train/unet_paper.yaml")
ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
args = ap.parse_args()
train(load_config(args.config, args.set))
