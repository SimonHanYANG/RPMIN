import math

import torch
from torch.nn import functional as F


# Pixel coordinates are (x, y), with x to the right and y down.
_ANGLES = {2: (0, 4), 4: (0, 2, 4, 6), 8: tuple(range(8))}
VIEW_MODES = ("difference", "random-direction", "shifted", "sobel", "scharr", "replicated")


def pixel_grid(height, width, device, dtype):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1)


def normalize_grid(grid, height, width):
    return (grid + 0.5) * grid.new_tensor((2 / width, 2 / height)) - 1


def displacements(batch, directions, mode, radius, generator=None):
    """Return B×D×2 offsets in LR pixels. A None radius samples U[0, 1)."""
    if directions == 0:
        return torch.empty(batch, 0, 2)
    if radius is not None and not 0 <= radius <= 1:
        raise ValueError("Fixed displacement radius must be in [0, 1].")
    device = generator.device if generator is not None else torch.device("cpu")
    draws = None
    if radius is None or mode == "random-direction":
        # Share per-image radii across input controls, including across batches.
        draws = torch.rand(batch, 9, generator=generator, device=device)
    radii = draws[:, :1, None] if radius is None else torch.full((batch, 1, 1), radius, device=device)
    if mode == "random-direction":
        angles = draws[:, 1:1 + directions] * (2 * math.pi)
        vectors = torch.stack((angles.cos(), angles.sin()), dim=-1)
    else:
        diagonal = math.sqrt(0.5)
        compass = torch.tensor(
            ((1, 0), (diagonal, diagonal), (0, 1), (-diagonal, diagonal),
             (-1, 0), (-diagonal, -diagonal), (0, -1), (diagonal, -diagonal)),
            device=device,
        )
        vectors = compass[list(_ANGLES[directions])]
    return radii * vectors


def directional_gradients(image, offsets, mode):
    if mode == "sobel":
        horizontal = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
        denominator = 8
    else:
        horizontal = ((-3, 0, 3), (-10, 0, 10), (-3, 0, 3))
        denominator = 32
    kernel = image.new_tensor(horizontal) / denominator
    filters = torch.stack((kernel, kernel.T))[:, None].repeat(image.shape[1], 1, 1, 1)
    gradients = F.conv2d(F.pad(image, (1,) * 4, mode="replicate"), filters, groups=image.shape[1])
    gradients = gradients.reshape(image.shape[0], image.shape[1], 2, *image.shape[-2:])
    return torch.einsum("bdk,bckhw->bdchw", offsets, gradients)


def build_views(image, offsets, mode="difference"):
    """Prepend the original image to the signed, directional views of Eq. (1)."""
    batch, channels, height, width = image.shape
    offsets = offsets.to(image)
    if offsets.ndim != 3 or offsets.shape[0] != batch or offsets.shape[-1] != 2:
        raise ValueError("Expected offsets of shape [batch, directions, 2].")
    directions = offsets.shape[1]
    if directions == 0:
        return image[:, None]
    if mode == "replicated":
        neighbors = image[:, None].expand(-1, directions, -1, -1, -1)
    elif mode in {"sobel", "scharr"}:
        neighbors = directional_gradients(image, offsets, mode)
    else:
        grid = pixel_grid(height, width, image.device, image.dtype)
        grid = normalize_grid(grid + offsets[:, :, None, None], height, width)
        shifted = F.grid_sample(
            image[:, None].expand(-1, directions, -1, -1, -1).reshape(-1, channels, height, width),
            grid.flatten(0, 1),
            mode="bicubic",
            padding_mode="border",
            align_corners=False,
        ).reshape(batch, directions, channels, height, width)
        neighbors = shifted if mode == "shifted" else shifted - image[:, None]
    return torch.cat((image[:, None], neighbors), dim=1)
