"""Segmenters: anything that turns a slice's T1w images into myocardium masks.

Interface:  segmenter(images (K,H,W), tis (K,), **ctx) -> masks (K,H,W) uint8
`ctx` may hold e.g. gt `masks` or `pixel_spacing`. Wrap any method this way --
a trained PyTorch net, nnU-Net / MONAI bundle predictor, a classical method, or
ground truth for testing the downstream stages in isolation.
"""
import numpy as np
import torch

from .data import normalize
from .device import resolve_device
from .io import pad_crop
from .registry import SEGMENTERS


@SEGMENTERS.register("torch")
class TorchSegmenter:
    """A checkpoint from scripts/train.py (any registered architecture)."""

    def __init__(self, checkpoint, device="auto", size=256, normalization="zscore", batch_size=16):
        from .models import load_checkpoint

        self.device = resolve_device(device)
        self.model, ck = load_checkpoint(checkpoint, self.device)
        self.size, self.normalization, self.bs = size, normalization, batch_size

    @torch.no_grad()
    def __call__(self, images, tis=None, **ctx):
        x = np.stack([normalize(im, self.normalization) for im in pad_crop(images, self.size)])
        x = torch.from_numpy(x)[:, None]
        out = [self.model(x[i:i + self.bs].to(self.device)).argmax(1).cpu() for i in range(0, len(x), self.bs)]
        return torch.cat(out).numpy().astype(np.uint8)


@SEGMENTERS.register("ground_truth")
class GroundTruthSegmenter:
    """Returns the masks stored with the data -- isolates alignment/fitting from segmentation errors."""

    def __init__(self, size=256):
        self.size = size

    def __call__(self, images, tis=None, masks=None, **ctx):
        if masks is None:
            raise ValueError("ground_truth segmenter needs `masks` in the input data")
        return pad_crop(np.asarray(masks, np.uint8), self.size)


@SEGMENTERS.register("cinema")
class CinemaSegmenter:
    """CineMA (ConvUNetR) cine-trained 3D segmenter applied per-frame to T1w images.

    Each T1-weighted image is fed independently as a single-slice 3D volume through
    a CineMA model pre-trained on UK Biobank cine and fine-tuned on ACDC or M&Ms.

    CineMA labels: 0=background, 1=RV, 2=myocardium, 3=LV.
    This segmenter extracts myocardium (label 2) to match the pipeline's binary mask
    interface.

    Requires: ``pip install --no-deps git+https://github.com/mathpluscode/CineMA``
    plus its dependencies (einops, hydra-core, omegaconf, monai, safetensors, timm,
    SimpleITK, huggingface_hub).
    """

    CINEMA_SIZE = 192

    def __init__(self, dataset="acdc", seed=0, device="auto", recenter=True,
                 target_spacing=None):
        self.dataset = dataset
        self.seed = seed
        self.device_spec = device
        self.recenter = recenter
        self.target_spacing = target_spacing
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model, self._device
        from cinema import ConvUNetR
        self._device = resolve_device(self.device_spec)
        view = "sax"
        self._model = ConvUNetR.from_finetuned(
            repo_id="mathpluscode/CineMA",
            model_filename=(
                f"finetuned/segmentation/{self.dataset}_{view}/"
                f"{self.dataset}_{view}_{self.seed}.safetensors"
            ),
            config_filename=(
                f"finetuned/segmentation/{self.dataset}_{view}/config.yaml"
            ),
        ).eval().to(self._device)
        return self._model, self._device

    @staticmethod
    def _clip_scale(img):
        lo, hi = np.percentile(img, (0.95, 99.5))
        v = np.clip(img, lo, hi)
        return ((v - v.min()) / (v.max() - v.min() + 1e-8)).astype(np.float32)

    def _predict_2d(self, img_hw, model, device):
        """(H, W) float image -> (H, W) uint8 4-class labels."""
        S = self.CINEMA_SIZE
        vol = self._clip_scale(img_hw.T[:, :, None])
        x = np.zeros((S, S, 16), np.float32)
        w, h = min(vol.shape[0], S), min(vol.shape[1], S)
        x[:w, :h, 0] = vol[:w, :h, 0]
        t = torch.from_numpy(x)[None, None].to(device)
        with torch.no_grad():
            logits = model({"sax": t})["sax"]
        return torch.argmax(logits, dim=1)[0, :w, :h, 0].cpu().numpy().astype(np.uint8).T

    @staticmethod
    def _crop_at(arr, y0, x0, size):
        H, W = arr.shape[-2:]
        out = np.zeros(arr.shape[:-2] + (size, size), arr.dtype)
        sy0, sy1 = max(0, y0), min(H, y0 + size)
        sx0, sx1 = max(0, x0), min(W, x0 + size)
        dy0, dx0 = max(0, -y0), max(0, -x0)
        h, w = sy1 - sy0, sx1 - sx0
        if h > 0 and w > 0:
            out[..., dy0:dy0 + h, dx0:dx0 + w] = arr[..., sy0:sy1, sx0:sx1]
        return out

    @staticmethod
    def _place_crop(crop, y0, x0, H, W):
        K, S = crop.shape[0], crop.shape[1]
        out = np.zeros((K, H, W), crop.dtype)
        sy0, sy1 = max(0, y0), min(H, y0 + S)
        sx0, sx1 = max(0, x0), min(W, x0 + S)
        dy0, dx0 = max(0, -y0), max(0, -x0)
        h, w = sy1 - sy0, sx1 - sx0
        if h > 0 and w > 0:
            out[:, sy0:sy1, sx0:sx1] = crop[:, dy0:dy0 + h, dx0:dx0 + w]
        return out

    @torch.no_grad()
    def __call__(self, images, tis=None, pixel_spacing=None, **ctx):
        model, device = self._ensure_model()
        K, H, W = images.shape
        S = self.CINEMA_SIZE

        sp = float(pixel_spacing[0]) if pixel_spacing is not None else None
        if self.target_spacing and sp and abs(sp - self.target_spacing) > 0.01:
            from scipy.ndimage import zoom as ndi_zoom
            factor = sp / self.target_spacing
            imgs = ndi_zoom(images, (1, factor, factor), order=1, mode="nearest")
        else:
            imgs = images

        _, Hr, Wr = imgs.shape
        y0, x0 = (Hr - S) // 2, (Wr - S) // 2
        imgs_crop = self._crop_at(imgs, y0, x0, S)
        pred = np.stack([self._predict_2d(im, model, device) for im in imgs_crop])

        if self.recenter:
            heart = np.argwhere((pred == 2) | (pred == 3))
            if len(heart) > 200:
                cy, cx = np.median(heart[:, 1:], axis=0)
                y0 += int(round(cy - S / 2))
                x0 += int(round(cx - S / 2))
                imgs_crop = self._crop_at(imgs, y0, x0, S)
                pred = np.stack([self._predict_2d(im, model, device)
                                 for im in imgs_crop])

        masks = self._place_crop((pred == 2).astype(np.uint8), y0, x0, Hr, Wr)

        if imgs is not images:
            from scipy.ndimage import zoom as ndi_zoom
            masks = (ndi_zoom(masks.astype(np.float32),
                              (1, H / Hr, W / Wr),
                              order=0, mode="nearest") > 0.5).astype(np.uint8)

        if masks.shape[-2:] != (H, W):
            out = np.zeros((K, H, W), np.uint8)
            mh, mw = min(masks.shape[1], H), min(masks.shape[2], W)
            out[:, :mh, :mw] = masks[:, :mh, :mw]
            masks = out

        return masks
