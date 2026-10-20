"""Small but trainable 2D/3D U-Net models; no pretrained weights required."""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim=3):
        super().__init__()
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.block = nn.Sequential(conv(in_channels, out_channels, 3, padding=1), nn.InstanceNorm3d(out_channels, affine=True) if dim == 3 else nn.InstanceNorm2d(out_channels, affine=True), nn.LeakyReLU(.1, inplace=True),
                                   conv(out_channels, out_channels, 3, padding=1), nn.LeakyReLU(.1, inplace=True))

    def forward(self, x):
        return self.block(x)


class MiniUNet(nn.Module):
    """Two-level U-Net, comparable design in 2D/3D. Input must have even spatial sizes."""
    def __init__(self, dim=3, base=8):
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.dim = dim
        pool = nn.MaxPool3d if dim == 3 else nn.MaxPool2d
        deconv = nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.enc1 = ConvBlock(1, base, dim)
        self.pool = pool(2)
        self.enc2 = ConvBlock(base, base*2, dim)
        self.up = deconv(base*2, base, 2, stride=2)
        self.dec = ConvBlock(base*2, base, dim)
        self.head = conv(base, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        up = self.up(b)
        if up.shape[2:] != a.shape[2:]:
            raise ValueError("Input spatial dimensions must be multiples of 2")
        return self.head(self.dec(torch.cat([a, up], dim=1)))


def dice_bce_loss(logits, mask):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, mask)
    proba = torch.sigmoid(logits)
    dimensions = tuple(range(1, logits.ndim))
    intersection = (proba * mask).sum(dimensions)
    denominator = proba.sum(dimensions) + mask.sum(dimensions)
    dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
    return bce + (1 - dice).mean()
