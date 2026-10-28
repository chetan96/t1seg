"""U-Net for myocardium segmentation of single T1-weighted images.

Follows Fahmy et al. (JCMR 2019): pre-activation "bottleneck" units
(BatchNorm -> ReLU -> Conv), dropout in every unit, channel doubling on
down-sampling / halving on up-sampling, 2-class softmax output.
With base=32, depth=4 the network has ~7.8M parameters (paper: ~9M).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """Two BN-ReLU-Conv units followed by dropout."""

    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.body(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=2, base=32, depth=4, dropout=0.2):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.stem = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.down = nn.ModuleList(
            [Bottleneck(chs[0], chs[0], dropout)]
            + [Bottleneck(chs[i - 1], chs[i], dropout) for i in range(1, depth + 1)]
        )
        self.up_conv = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2) for i in range(depth, 0, -1)]
        )
        self.up = nn.ModuleList(
            [Bottleneck(2 * chs[i - 1], chs[i - 1], dropout) for i in range(depth, 0, -1)]
        )
        self.head = nn.Sequential(nn.BatchNorm2d(chs[0]), nn.ReLU(inplace=True),
                                  nn.Conv2d(chs[0], n_classes, 1))

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i, blk in enumerate(self.down):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = blk(x)
            skips.append(x)
        skips.pop()  # deepest feature map is x itself
        for upc, blk in zip(self.up_conv, self.up):
            x = upc(x)
            x = blk(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)  # logits (B, n_classes, H, W)


if __name__ == "__main__":
    m = UNet()
    print(f"params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
    print(m(torch.zeros(2, 1, 256, 256)).shape)
