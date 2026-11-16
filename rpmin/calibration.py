from dataclasses import dataclass
import json
import math
from pathlib import Path

from torch.nn import functional as F

from .views import normalize_grid, pixel_grid


@dataclass(frozen=True)
class CameraCalibration:
    """Fixed LR translation and optional tone correction fitted on training pairs."""

    shift_lr: tuple[float, float] = (0.0, 0.0)
    tone_coefficients: tuple[tuple[float, ...], ...] | None = None
    tone_strength: float = 0.0

    def __post_init__(self):
        if len(self.shift_lr) != 2 or not all(math.isfinite(value) for value in self.shift_lr):
            raise ValueError("Calibration shift must contain two finite LR pixel offsets.")
        if not 0 <= self.tone_strength <= 1:
            raise ValueError("Tone strength must be between zero and one.")
        if self.tone_coefficients is not None and (
            len(self.tone_coefficients) != 3 or any(
                len(row) != 6 or not all(math.isfinite(value) for value in row) for row in self.tone_coefficients
            )
        ):
            raise ValueError("Tone calibration needs six finite coefficients per RGB channel.")
        if self.tone_strength and self.tone_coefficients is None:
            raise ValueError("Nonzero tone strength requires coefficients.")

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as stream:
            inputs = json.load(stream)
        return cls(**inputs)

    def align(self, image):
        if not any(self.shift_lr):
            return image
        height, width = image.shape[-2:]
        grid = pixel_grid(height, width, image.device, image.dtype)
        grid = normalize_grid(grid - image.new_tensor(self.shift_lr), height, width)
        return F.grid_sample(
            image, grid[None].expand(image.shape[0], -1, -1, -1),
            mode="bilinear", padding_mode="reflection", align_corners=False,
        )

    def correct_tone(self, prediction, image):
        if not self.tone_strength:
            return prediction
        size = prediction.shape[-2:]
        base = F.interpolate(image, size=size, mode="bicubic", align_corners=False)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        bias, gain, quadratic, cubic, mean_bias, mean_gain = (
            base.new_tensor(self.tone_coefficients).T[:, None, :, None, None]
        )
        corrected = bias + base * (gain + base * (quadratic + base * cubic)) + mean * (mean_bias + mean_gain * base)
        return prediction + self.tone_strength * (corrected - base)
