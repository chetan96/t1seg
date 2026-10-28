"""Adapters for third-party model zoos. Imported lazily, so they're optional dependencies.

    model: {name: smp, arch: Unet, encoder_name: resnet34, encoder_weights: imagenet}
    model: {name: monai, net: SwinUNETR, img_size: [256, 256], spatial_dims: 2, feature_size: 24}
    model: {name: torchvision_deeplab, backbone: resnet50}
"""
import torch.nn as nn

from ..registry import MODELS


@MODELS.register("smp")
def smp_model(arch="Unet", encoder_name="resnet34", encoder_weights=None, in_channels=1, n_classes=2, **kw):
    """segmentation_models_pytorch: Unet, UnetPlusPlus, FPN, DeepLabV3Plus, MAnet, Segformer, ... with
    100+ (optionally ImageNet-pretrained) encoders.  pip install segmentation-models-pytorch"""
    import segmentation_models_pytorch as smp

    return getattr(smp, arch)(encoder_name=encoder_name, encoder_weights=encoder_weights,
                              in_channels=in_channels, classes=n_classes, **kw)


@MODELS.register("monai")
def monai_model(net="UNet", in_channels=1, n_classes=2, spatial_dims=2, **kw):
    """Any monai.networks.nets class (UNet, AttentionUnet, SwinUNETR, UNETR, DynUNet, ...).
    pip install monai"""
    from monai.networks import nets

    return getattr(nets, net)(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=n_classes, **kw)


class _OutKey(nn.Module):
    """torchvision segmentation models return {'out': logits}; unwrap to a tensor."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["out"]


@MODELS.register("torchvision_deeplab")
def torchvision_deeplab(backbone="resnet50", in_channels=1, n_classes=2, **kw):
    from torchvision.models import segmentation as tvs

    m = getattr(tvs, f"deeplabv3_{backbone}")(weights=None, weights_backbone=None, num_classes=n_classes, **kw)
    if in_channels != 3:
        old = m.backbone.conv1
        m.backbone.conv1 = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride,
                                     old.padding, bias=False)
    return _OutKey(m)
