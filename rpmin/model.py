from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import DeformConv2d

from .views import build_views, displacements, normalize_grid, pixel_grid


@dataclass(frozen=True)
class ModelConfig:
    scale: int = 4
    width: int = 32
    directions: int = 8
    view_mode: str = "difference"
    amfm: bool = True
    rpum: bool = True
    deformable: bool = True
    radius: float | None = None
    detail_head: bool = False

    def __post_init__(self):
        if self.scale not in (2, 4) or self.width < 1:
            raise ValueError("Use scale 2 or 4 and a positive width.")
        if self.directions != 8 or self.view_mode != "difference" or not all((self.amfm, self.rpum, self.deformable)):
            raise ValueError("RP-MIN uses eight directional differences, AMFM, RPUM and deformable convolutions.")
        if self.radius is not None and not 0 <= self.radius <= 1:
            raise ValueError("Fixed radius must be in [0, 1].")


class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DeformConv2d(in_channels, out_channels, 3, padding=1)
        self.offsets = nn.Conv2d(in_channels, 18, 3, padding=1)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    def forward(self, features):
        return self.conv(features, self.offsets(features))


class AMFM(nn.Module):
    def __init__(self, width, views):
        super().__init__()
        self.embedding = AdaptiveConv(width, width)
        self.distinctiveness = nn.Conv2d((views - 1) * width, width, 3, padding=1)
        self.fuse = nn.Conv2d(2 * views * width, width, 1)

    def forward(self, features):
        batch, views, channels, height, width = features.shape
        embeddings = self.embedding(features.flatten(0, 1)).reshape(batch, views, channels, height, width)
        reference = embeddings[:, 0]
        neighbors = embeddings[:, 1:]
        sce = (neighbors * reference[:, None]).sum(dim=1).sigmoid()
        difference = (neighbors - reference[:, None]).abs().flatten(1, 2)
        sda = self.distinctiveness(difference).sigmoid()
        adjusted = torch.cat(
            ((features * sce[:, None]).flatten(1, 2), (features * sda[:, None]).flatten(1, 2)),
            dim=1,
        )
        return self.fuse(adjusted), {"sce": sce, "sda": sda}


def feature_block(in_channels, out_channels):
    return nn.Sequential(
        AdaptiveConv(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1),
    )


class EncoderDecoder(nn.Module):
    """Two down/up stages with same-resolution skips, following Fig. 2."""

    def __init__(self, width):
        super().__init__()
        self.encoder1 = feature_block(width, width)
        self.encoder2 = feature_block(width * 2, width * 2)
        self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(width * 4, width * 4, 3, padding=1), nn.LeakyReLU(0.1)
        )
        self.decoder2 = feature_block(width * 6, width * 2)
        self.decoder1 = feature_block(width * 3, width)

    def forward(self, fused):
        fine = self.encoder1(fused)
        coarse = self.encoder2(self.down1(fine))
        bottleneck = self.bottleneck(self.down2(coarse))
        upper = F.interpolate(bottleneck, size=coarse.shape[-2:], mode="bilinear", align_corners=False)
        upper = self.decoder2(torch.cat((upper, coarse), dim=1))
        upper = F.interpolate(upper, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        return self.decoder1(torch.cat((upper, fine), dim=1)) + fused


class RPUM(nn.Module):
    """Predict a displacement in LR pixels and modulate it per subpixel phase."""

    def __init__(self, width, scale):
        super().__init__()
        self.scale = scale
        self.offsets = nn.Conv2d(width, 2, 3, padding=1)
        self.phase = nn.Conv2d(width, 2 * scale**2, 3, padding=1)
        self.to_rgb = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)
        # Start at the bicubic skip so the untrained residual cannot tint the image.
        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    def forward(self, features):
        scale = self.scale
        height, width = features.shape[-2:]
        offsets = self.offsets(features)[:, :, None] * (2 * self.phase(features).sigmoid()).reshape(
            -1, 2, scale**2, height, width
        )
        offsets = F.pixel_shuffle(offsets.flatten(1, 2), scale).permute(0, 2, 3, 1)
        base = pixel_grid(height * scale, width * scale, features.device, features.dtype)
        base = (base + 0.5) / scale - 0.5
        sampling_grid = normalize_grid(base + offsets, height, width)
        sampled = F.grid_sample(
            features, sampling_grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return self.to_rgb(sampled)


class DetailHead(nn.Module):
    """Reconstruct a separate residual for each HR phase on the LR feature grid."""

    def __init__(self, width, scale):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(0.1),
        )
        self.to_phases = nn.Conv2d(width, 3 * scale**2, 3, padding=1)
        self.scale = scale
        nn.init.zeros_(self.to_phases.weight)
        nn.init.zeros_(self.to_phases.bias)

    def forward(self, features):
        return F.pixel_shuffle(self.to_phases(features + self.refine(features)), self.scale)


class RPMIN(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.checkpoint_epoch: int | None = None
        self.stem = nn.Sequential(nn.Conv2d(3, config.width, 3, padding=1), nn.LeakyReLU(0.1))
        self.fusion = AMFM(config.width, config.directions + 1)
        self.backbone = EncoderDecoder(config.width)
        self.upsample = RPUM(config.width, config.scale)
        self.detail = DetailHead(config.width, config.scale) if config.detail_head else None

    def forward(self, image, *, generator=None, offsets=None, return_maps=False):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 4:
            raise ValueError("Expected B×3×H×W RGB input with spatial dimensions at least 4.")
        if offsets is None:
            offsets = displacements(image.shape[0], self.config.radius, generator)
        if offsets.shape != (image.shape[0], self.config.directions, 2):
            raise ValueError("Offsets must match the model's batch size and number of directions.")
        views = build_views(image, offsets)
        features = self.stem(views.flatten(0, 1)).reshape(
            image.shape[0], self.config.directions + 1, self.config.width, *image.shape[-2:]
        )
        fused, maps = self.fusion(features)
        decoded = self.backbone(fused)
        residual = self.upsample(decoded)
        if self.detail is not None:
            residual = residual + self.detail(decoded)
        base = F.interpolate(image, scale_factor=self.config.scale, mode="bicubic", align_corners=False)
        result = base + residual
        return (result, maps) if return_maps else result
