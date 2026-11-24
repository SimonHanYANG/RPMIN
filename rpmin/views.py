import math

import torch
from torch.nn import functional as F


def pixel_grid(height, width, device, dtype):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1)


def normalize_grid(grid, height, width):
    return (grid + 0.5) * grid.new_tensor((2 / width, 2 / height)) - 1


def displacements(batch, radius, generator=None):
    """Return eight offsets in LR pixels per image. A None radius samples U[0, 1)."""
    if radius is not None and not 0 <= radius <= 1:
        raise ValueError("Fixed displacement radius must be in [0, 1].")
    device = generator.device if generator is not None else torch.device("cpu")
    if radius is None:
        # Keep the archived draw stride so a fixed seed reproduces the same radii.
        radii = torch.rand(batch, 9, generator=generator, device=device)[:, :1, None]
    else:
        radii = torch.full((batch, 1, 1), radius, device=device)
    diagonal = math.sqrt(0.5)
    compass = torch.tensor(
        ((1, 0), (diagonal, diagonal), (0, 1), (-diagonal, diagonal),
         (-1, 0), (-diagonal, -diagonal), (0, -1), (diagonal, -diagonal)),
        device=device,
    )
    return radii * compass


def build_views(image, offsets):
    """Prepend the original image to the signed, directional views of Eq. (1)."""
    batch, channels, height, width = image.shape
    offsets = offsets.to(image)
    if offsets.ndim != 3 or offsets.shape[0] != batch or offsets.shape[-1] != 2:
        raise ValueError("Expected offsets of shape [batch, directions, 2].")
    directions = offsets.shape[1]
    if directions == 0:
        raise ValueError("At least one direction is required.")
    grid = pixel_grid(height, width, image.device, image.dtype)
    grid = normalize_grid(grid + offsets[:, :, None, None], height, width)
    shifted = F.grid_sample(
        image[:, None].expand(-1, directions, -1, -1, -1).reshape(-1, channels, height, width),
        grid.flatten(0, 1),
        mode="bicubic",
        padding_mode="border",
        align_corners=False,
    ).reshape(batch, directions, channels, height, width)
    return torch.cat((image[:, None], shifted - image[:, None]), dim=1)
