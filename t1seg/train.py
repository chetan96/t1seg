"""Config-driven training loop for any registered model / loss / dataset."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .device import loader_kwargs, resolve_device
from .losses import build_loss
from .models import build_model, init_weights_from, save_checkpoint
from .registry import DATASETS


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


def build_optimizer(params, cfg):
    cfg = dict(cfg)
    name = cfg.pop("name", "Adam")
    return getattr(torch.optim, name)(params, **cfg)


def build_scheduler(opt, cfg):
    if not cfg:
        return None
    cfg = dict(cfg)
    return getattr(torch.optim.lr_scheduler, cfg.pop("name"))(opt, **cfg)


def train(cfg):
    dc, tc = cfg["data"], cfg["train"]
    device = resolve_device(cfg.get("device", "auto"))
    torch.manual_seed(tc.get("seed", 0))

    train_ds = DATASETS.build(dc["train"])
    val_ds = DATASETS.build(dc["val"])
    lk = dict(num_workers=tc.get("num_workers", 0), **loader_kwargs(device))
    train_dl = DataLoader(train_ds, tc["batch_size"], shuffle=True, **lk)
    val_dl = DataLoader(val_ds, tc["batch_size"], **lk)

    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device} model={cfg['model']['name']} ({n_params:.2f}M) "
          f"train_imgs={len(train_ds)} val_imgs={len(val_ds)}")
    if tc.get("init_from"):
        n, tot = init_weights_from(model, tc["init_from"])
        print(f"transfer learning: loaded {n}/{tot} tensors from {tc['init_from']}")

    loss_fn = build_loss(cfg.get("loss", "ce")).to(device)
    opt = build_optimizer(model.parameters(), cfg.get("optimizer", {"name": "Adam", "lr": 1e-3}))
    sched = build_scheduler(opt, cfg.get("scheduler"))

    out = Path(tc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    best, log = -1.0, []
    for ep in range(tc["epochs"]):
        model.train()
        losses = []
        for x, y in tqdm(train_dl, desc=f"epoch {ep}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if sched:
            sched.step()
        d_mean, d_sd = evaluate(model, val_dl, device)
        log.append(dict(epoch=ep, loss=float(np.mean(losses)), val_dice=d_mean, val_dice_sd=d_sd))
        print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  val DSC {d_mean:.3f} ± {d_sd:.3f}", flush=True)
        meta = dict(epoch=ep, val_dice=d_mean, config=cfg)
        save_checkpoint(out / "last.pt", model, cfg["model"], cfg.get("plugins"), **meta)
        if d_mean > best:
            best = d_mean
            save_checkpoint(out / "best.pt", model, cfg["model"], cfg.get("plugins"), **meta)
        (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"best val DSC {best:.3f} -> {out / 'best.pt'}")
    return out / "best.pt"
