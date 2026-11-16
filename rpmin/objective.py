import torch
from torch import nn
from torch.nn import functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, pixel_weight=0.9, frequency_weight=0.1, phase="angle", mse_weight=0.0, ssim_weight=0.0):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.frequency_weight = frequency_weight
        self.phase = phase
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight

    def forward(self, sr, hr):
        error = (sr - hr).square()
        loss = self.pixel_weight * (error + 1e-12).sqrt().mean()
        if self.mse_weight:
            loss = loss + self.mse_weight * error.mean()
        if self.ssim_weight:
            similarity, _ = ssim_map(sr.float(), hr.float())
            loss = loss + self.ssim_weight * (1 - similarity.mean())
        if not self.frequency_weight:
            return loss
        sr_fft, hr_fft = (torch.fft.fft2(x.float(), norm="ortho") for x in (sr, hr))
        sr_magnitude, hr_magnitude = sr_fft.abs(), hr_fft.abs()
        amplitude = (sr_magnitude - hr_magnitude).square().mean()
        if self.phase == "energy-weighted":
            # Phase is poorly determined where either spectrum has little energy.
            # Detach the confidence so it does not directly reward shrinking amplitudes.
            eps = torch.finfo(sr_magnitude.dtype).eps
            confidence = torch.minimum(sr_magnitude.detach(), hr_magnitude).square()
            sr_unit = sr_fft / sr_magnitude.clamp_min(eps)
            hr_unit = hr_fft / hr_magnitude.clamp_min(eps)
            phase_loss = (confidence * (sr_unit - hr_unit).abs().square()).mean()
        elif self.phase == "circular":
            delta = sr_fft.angle() - hr_fft.angle()
            phase_loss = (2 - 2 * delta.cos()).mean()
        else:
            phase_loss = (sr_fft.angle() - hr_fft.angle()).square().mean()
        return loss + self.frequency_weight * (amplitude + phase_loss)


def image_metrics(sr, hr, border=0):
    """Per-image RGB PSNR and Gaussian-window SSIM on [0, 1]."""
    if sr.shape != hr.shape:
        raise ValueError(f"Image shapes differ: {sr.shape} vs {hr.shape}.")
    if border < 0 or 2 * border >= min(hr.shape[-2:]):
        raise ValueError("Metric border must leave a nonempty image.")
    if border:
        sr, hr = (x[..., border:-border, border:-border] for x in (sr, hr))
    sr, hr = sr.float().clamp(0, 1), hr.float()
    error = (sr - hr).square().mean(dim=(1, 2, 3))
    psnr = -10 * error.clamp_min(1e-12).log10()
    similarity, _ = ssim_map(sr, hr)
    return psnr, similarity.mean(dim=(1, 2, 3))


def ssim_map(sr, hr):
    window = min(11, *sr.shape[-2:])
    window = window - (1 - window % 2)
    coord = torch.arange(window, device=sr.device, dtype=sr.dtype) - window // 2
    gaussian = (-coord.square() / (2 * 1.5**2)).exp()
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(sr.shape[1], 1, window, window)

    def average(image):
        return F.conv2d(image, kernel, groups=sr.shape[1])

    mu_sr, mu_hr = average(sr), average(hr)
    var_sr = average(sr.square()) - mu_sr.square()
    var_hr = average(hr.square()) - mu_hr.square()
    covariance = average(sr * hr) - mu_sr * mu_hr
    luminance = (2 * mu_sr * mu_hr + 0.01**2) / (mu_sr.square() + mu_hr.square() + 0.01**2)
    structure = (2 * covariance + 0.03**2) / (var_sr + var_hr + 0.03**2)
    return luminance * structure, window


def grid_metrics(sr, hr, scale, patch, stride_x, stride_y, origin=1, border=0):
    """Score fixed LR-grid windows from one already reconstructed full field."""
    if sr.shape != hr.shape or sr.ndim != 4 or sr.shape[0] != 1:
        raise ValueError("Patch scoring needs one matching BCHW prediction and reference.")
    if min(scale, patch, stride_x, stride_y) < 1 or min(origin, border) < 0:
        raise ValueError("Patch size, stride and scale must be positive, with nonnegative origin and border.")
    h, w = sr.shape[-2:]
    if h % scale or w % scale or patch * scale - 2 * border < 11:
        raise ValueError("Patch scoring requires a scale-matched image and at least 11 scored HR pixels per side.")
    xs = range(origin, w // scale - origin - patch + 1, stride_x)
    ys = range(origin, h // scale - origin - patch + 1, stride_y)
    points = [(x, y) for y in ys for x in xs]
    if not points:
        raise ValueError("No valid patches for this field and grid.")
    sr, hr = sr.float().clamp(0, 1), hr.float()
    error = (sr - hr).square().mean(dim=1)[0]
    similarity, window = ssim_map(sr, hr)
    coords = torch.tensor(points, device=sr.device)
    x, y = (coords * scale + border).T
    length = patch * scale - 2 * border

    def neighborhood_means(values, side):
        integral = F.pad(values.cumsum(0, dtype=torch.float64).cumsum(1), (1, 0, 1, 0))
        total = integral[y + side, x + side] - integral[y, x + side] - integral[y + side, x] + integral[y, x]
        return total / side**2

    psnr = -10 * neighborhood_means(error, length).clamp_min(1e-12).log10()
    ssim = neighborhood_means(similarity.mean(dim=1)[0], length - window + 1)
    return points, psnr, ssim
