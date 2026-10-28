"""Train the myocardium U-Net.

    python scripts/train.py -c configs/default.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from t1seg.data import T1wSegDataset
from t1seg.model import UNet
from t1seg.pipeline import get_device


def soft_dice_loss(logits, target):
    p = logits.softmax(1)[:, 1]
    t = target.float()
    inter = (p * t).sum((1, 2))
    return (1 - (2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices = []
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        inter = ((pred == 1) & (y == 1)).sum((1, 2)).float()
        tot = (pred == 1).sum((1, 2)) + (y == 1).sum((1, 2))
        dices += (2 * inter / tot.clamp(min=1)).tolist()
    return float(np.mean(dices)), float(np.std(dices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, help="override config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    dc, mc, tc = cfg["data"], cfg["model"], cfg["train"]

    device = get_device()
    torch.manual_seed(0)
    train_ds = T1wSegDataset.from_dir(dc["train_dir"], size=dc["size"], augment=dc["augment"])
    val_ds = T1wSegDataset.from_dir(dc["val_dir"], size=dc["size"])
    train_dl = DataLoader(train_ds, tc["batch_size"], shuffle=True, num_workers=tc["num_workers"])
    val_dl = DataLoader(val_ds, tc["batch_size"], num_workers=tc["num_workers"])
    print(f"device={device} train_imgs={len(train_ds)} val_imgs={len(val_ds)}")

    model = UNet(**mc).to(device)
    if tc.get("init_from"):
        model.load_state_dict(torch.load(tc["init_from"], map_location=device)["model"])
        print(f"initialised from {tc['init_from']}")
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, tc["lr_gamma"])

    out = Path(tc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    for ep in range(tc["epochs"]):
        model.train()
        losses = []
        for x, y in tqdm(train_dl, desc=f"epoch {ep}", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + tc["dice_weight"] * soft_dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        d_mean, d_sd = evaluate(model, val_dl, device)
        log.append(dict(epoch=ep, loss=float(np.mean(losses)), val_dice=d_mean, val_dice_sd=d_sd))
        print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  val DSC {d_mean:.3f} ± {d_sd:.3f}")
        ck = dict(model=model.state_dict(), model_kwargs=mc, epoch=ep, val_dice=d_mean, config=cfg)
        torch.save(ck, out / "last.pt")
        if d_mean > best:
            best = d_mean
            torch.save(ck, out / "best.pt")
        json.dump(log, open(out / "log.json", "w"), indent=1)
    print(f"best val DSC {best:.3f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
