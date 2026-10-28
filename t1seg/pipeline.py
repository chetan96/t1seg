"""End-to-end per-slice pipeline (paper Fig. 1):

  T1w images --FCN--> masks --clean/QC/refine--> polar resampling (per image)
  --pixel-wise fit--> polar T1 map --inverse polar--> Cartesian T1 map --> mid-wall stats
"""
import numpy as np
import torch

from .analysis import T1_RANGE_1P5T, midwall_mask, slice_stats
from .data import normalize
from .io import pad_crop
from .model import UNet
from .polar import C_DEPTH, N_ANGLES, from_polar, to_polar
from .postprocess import postprocess_series
from .t1fit import fit_map

MIN_GOOD_IMAGES = 8  # paper: reconstruct only if >= 8 of 11 images segmented OK


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt, device=None):
    device = device or get_device()
    ck = torch.load(ckpt, map_location=device)
    model = UNet(**ck.get("model_kwargs", {})).to(device).eval()
    model.load_state_dict(ck["model"])
    return model


@torch.no_grad()
def segment(model, images, size=256):
    dev = next(model.parameters()).device
    x = np.stack([normalize(im) for im in pad_crop(images, size)])
    logits = model(torch.from_numpy(x)[:, None].to(dev))
    return logits.argmax(1).cpu().numpy().astype(np.uint8)


def run_slice(images, tis, masks=None, model=None, pixel_spacing=(1.4, 1.4), fit_model="2p",
              min_good=MIN_GOOD_IMAGES, t1_range=T1_RANGE_1P5T, reference="shortest_ti"):
    """Process one slice. Pass `masks` to bypass the FCN (e.g. to test with manual masks).

    Returns dict with masks, ok flags, polar map, Cartesian T1 map, mid-wall mask, stats.
    """
    images = pad_crop(np.asarray(images, np.float32), 256)
    tis = np.asarray(tis, float)
    raw = segment(model, images) if masks is None else pad_crop(np.asarray(masks, np.uint8), 256)
    masks, ok = postprocess_series(raw, tis, pixel_spacing)
    out = dict(raw_masks=raw, masks=masks, ok=ok, images=images, tis=tis)
    if ok.sum() < min_good:
        out["status"] = f"skipped: only {ok.sum()}/{len(ok)} usable masks"
        return out

    idx = np.where(ok)[0]
    polar = np.stack([to_polar(images[k], masks[k], N_ANGLES, C_DEPTH)[0] for k in idx])
    polar_t1 = fit_map(polar, tis[idx], fit_model)

    good_tis = np.where(np.isfinite(tis[idx]), tis[idx], np.inf)
    k_ref = idx[np.argmin(good_tis)] if reference == "shortest_ti" else idx[np.argmax(good_tis)]
    t1_map = from_polar(polar_t1, masks[k_ref])
    mid = midwall_mask(masks[k_ref])
    out.update(status="ok", polar=polar, polar_t1=polar_t1, t1_map=t1_map, ref_index=int(k_ref),
               midwall=mid, stats=slice_stats(t1_map, mid, t1_range))
    return out
