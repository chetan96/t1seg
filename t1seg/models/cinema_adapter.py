"""CineMA ConvUNetR adapter for our 2D training loop.

Wraps the 3D CineMA model (4-class, dict-based I/O) into the standard
(B, 1, H, W) -> (B, 2, H, W) interface expected by train.py / finetune_eval.py.

Usage in configs or CLI:
    model: {name: cinema_ft, dataset: acdc, seed: 0}

The first time the model is built it downloads ~510 MB from HuggingFace.
"""
import torch
import torch.nn as nn

from ..registry import MODELS


class CinemaAdapter(nn.Module):
    """Adapt CineMA's 3D 4-class segmenter for 2D 2-class training.

    CineMA: {"sax": (B, 1, W, H, 16)} -> {"sax": (B, 4, W, H, 16)}
    This:   (B, 1, H, W)              -> (B, 2, H, W)

    Label mapping: CineMA class 2 (myocardium) -> class 1, rest -> class 0.
    A learned 1x1 conv head replaces the original 4-class output.
    """
    CINEMA_SIZE = 192

    def __init__(self, backbone, head, cinema_size=192):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.cinema_size = cinema_size

    def forward(self, x):
        B, C, H, W = x.shape
        S = self.cinema_size
        vol = torch.zeros(B, C, S, S, 16, device=x.device, dtype=x.dtype)
        h, w = min(H, S), min(W, S)
        vol[:, :, :h, :w, 0] = x[:, :, :h, :w]
        logits_3d = self.backbone({"sax": vol})["sax"]
        logits_2d = logits_3d[:, :, :h, :w, 0]
        out = self.head(logits_2d)
        if h < H or w < W:
            padded = torch.zeros(B, 2, H, W, device=x.device, dtype=x.dtype)
            padded[:, :, :h, :w] = out
            return padded
        return out


def _init_head_from_cinema(head, bias_myo_class=2):
    """Initialize the 2-class head so class 1 starts near CineMA's myocardium logit."""
    with torch.no_grad():
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        head.weight[0, 0] = 1.0  # bg <- CineMA bg
        head.weight[0, 1] = 0.5  # bg gets some RV signal
        head.weight[0, 3] = 0.5  # bg gets some LV signal
        head.weight[1, bias_myo_class] = 1.0  # myo <- CineMA myo


@MODELS.register("cinema_ft")
def cinema_finetune_model(dataset="acdc", seed=0, freeze_encoder=False, n_classes=2, **kw):
    """Load a CineMA segmentation model wrapped for 2D 2-class fine-tuning."""
    from cinema import ConvUNetR

    model = ConvUNetR.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=(
            f"finetuned/segmentation/{dataset}_sax/"
            f"{dataset}_sax_{seed}.safetensors"
        ),
        config_filename=(
            f"finetuned/segmentation/{dataset}_sax/config.yaml"
        ),
    )

    out_chans = 4
    head = nn.Conv2d(out_chans, n_classes, kernel_size=1)
    _init_head_from_cinema(head)

    if freeze_encoder:
        for name, p in model.named_parameters():
            if "enc" in name:
                p.requires_grad = False

    return CinemaAdapter(backbone=model, head=head, cinema_size=192)
