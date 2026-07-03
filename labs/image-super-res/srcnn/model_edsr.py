import math
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Module):
    def __init__(self, scale, n_feats):
        super().__init__()
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                layers.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1))
                layers.append(nn.PixelShuffle(2))
        else:
            layers.append(nn.Conv2d(n_feats, scale * scale * n_feats, 3, padding=1))
            layers.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class EDSR(nn.Module):
    def __init__(self, scale=4, n_feats=256, n_resblocks=32, res_scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)

        body = [ResBlock(n_feats, res_scale) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*body)

        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, 3, 3, padding=1),
        )

    def forward(self, x):
        x = self.head(x)
        x = x + self.body(x)
        x = self.tail(x)
        return x
