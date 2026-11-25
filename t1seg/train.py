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


def train(cfg, model=None):
    dc, tc = cfg["data"], cfg["train"]
    device = resolve_device(cfg.get("device", "auto"))
    torch.manual_seed(tc.get("seed", 0))

    train_ds = DATASETS.build(dc["train"])
    val_ds = DATASETS.build(dc["val"])
    lk = dict(num_workers=tc.get("num_workers", 0), **loader_kwargs(device))
    train_dl = DataLoader(train_ds, tc["batch_size"], shuffle=True, **lk)
    val_dl = DataLoader(val_ds, tc["batch_size"], **lk)

    if model is None:
        model = build_model(cfg["model"]).to(device)
        if tc.get("init_from"):
            n, tot = init_weights_from(model, tc["init_from"])
            print(f"transfer learning: loaded {n}/{tot} tensors from {tc['init_from']}")
    else:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device} model={cfg['model']['name']} ({n_params:.2f}M) "
          f"train_imgs={len(train_ds)} val_imgs={len(val_ds)}")

    loss_fn = build_loss(cfg.get("loss", "ce")).to(device)
    opt = build_optimizer(model.parameters(), cfg.get("optimizer", {"name": "Adam", "lr": 1e-3}))
    sched = build_scheduler(opt, cfg.get("scheduler"))

    start_epoch = 0
    best = -1.0
    log = []

    if tc.get("resume_from"):
        ck = torch.load(tc["resume_from"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck and sched is not None:
            sched.load_state_dict(ck["scheduler"])
        start_epoch = ck.get("epoch", -1) + 1
        best = ck.get("best_val_dice", ck.get("val_dice", -1.0))
        log_path = Path(tc["out_dir"]) / "log.json"
        if log_path.exists():
            log = json.loads(log_path.read_text())
        print(f"resuming from epoch {start_epoch}, best val DSC so far {best:.3f}")

    out = Path(tc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    patience = tc.get("patience", 0)
    epochs_no_improve = 0

    for ep in range(start_epoch, tc["epochs"]):
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
        meta = dict(epoch=ep, val_dice=d_mean, best_val_dice=max(best, d_mean), config=cfg)
        save_checkpoint(out / "last.pt", model, cfg["model"], cfg.get("plugins"),
                        optimizer=opt, scheduler=sched, **meta)
        if d_mean > best:
            best = d_mean
            epochs_no_improve = 0
            save_checkpoint(out / "best.pt", model, cfg["model"], cfg.get("plugins"),
                            optimizer=opt, scheduler=sched, **meta)
        else:
            epochs_no_improve += 1
        (out / "log.json").write_text(json.dumps(log, indent=1))
        if patience and epochs_no_improve >= patience:
            print(f"early stopping: no val DSC improvement for {patience} epochs")
            break
    print(f"best val DSC {best:.3f} -> {out / 'best.pt'}")
    return out / "best.pt", model
