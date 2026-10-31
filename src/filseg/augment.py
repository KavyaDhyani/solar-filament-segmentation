"""Network inputs and GPU-side augmentation.

Input channels (all derived from one grayscale image):
0. intensity normalised by the disk's mean and std;
1. limb-darkening-corrected contrast, (c - 1) / 0.1;
2. distance from the disk centre in disk radii, (rho - 0.5) / 0.3 (a position prior: experts
   do not annotate near the limb).

Geometric augmentation moves all channels and the target together, so the rho channel stays
consistent with the image.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .prep import CON_LO, CON_SPAN

FULL = 2048


def rho_map(cx: torch.Tensor, cy: torch.Tensor, r: torch.Tensor, size: int) -> torch.Tensor:
    """(B, size, size) distance from the disk centre / radius; disk parameters in 2048-pixel coordinates."""
    s = FULL / size
    coords = (torch.arange(size, device=cx.device, dtype=torch.float32) + 0.5) * s - 0.5
    yy, xx = coords.view(1, size, 1), coords.view(1, 1, size)
    return torch.hypot(xx - cx.view(-1, 1, 1), yy - cy.view(-1, 1, 1)) / r.view(-1, 1, 1)


def make_input(img_u8: torch.Tensor, con_u8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    ch0 = (img_u8.float() - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)
    ch1 = (CON_LO + con_u8.float() / 255.0 * CON_SPAN - 1.0) / 0.1
    ch2 = (rho.clamp(max=1.5) - 0.5) / 0.3
    return torch.stack([ch0, ch1, ch2], dim=1)


def d4(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, k, dims=(-2, -1)) if k else x
    return x.flip(-1) if flip else x


def d4_inverse(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = x.flip(-1) if flip else x
    return torch.rot90(x, -k, dims=(-2, -1)) if k else x


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(2.5 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def augment(x: torch.Tensor, y: torch.Tensor, gen: torch.Generator, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (B, 3, S, S) inputs, y: (B, 1, S, S) soft target. Returns augmented copies."""
    b = x.shape[0]
    u = lambda *shape: torch.rand(*shape, generator=gen, device="cpu")  # noqa: E731
    xs, ys = [], []
    for i in range(b):  # dihedral group: exact, no resampling
        k, flip = int(u(1) * 4) % 4, bool(u(1) < 0.5)
        xs.append(d4(x[i], k, flip))
        ys.append(d4(y[i], k, flip))
    x, y = torch.stack(xs), torch.stack(ys)

    if cfg.get("p_affine", 0) > 0:  # small rotation, scale and shift
        apply = u(b) < cfg["p_affine"]
        ang = (u(b) * 2 - 1) * math.radians(cfg.get("max_rot", 15))
        scale = 1 + (u(b) * 2 - 1) * cfg.get("max_scale", 0.08)
        shift = (u(b, 2) * 2 - 1) * cfg.get("max_shift", 0.03)
        ang, scale, shift = ang * apply, torch.where(apply, scale, torch.ones_like(scale)), shift * apply[:, None]
        theta = torch.zeros(b, 2, 3)
        theta[:, 0, 0] = torch.cos(ang) / scale
        theta[:, 0, 1] = -torch.sin(ang) / scale
        theta[:, 1, 0] = torch.sin(ang) / scale
        theta[:, 1, 1] = torch.cos(ang) / scale
        theta[:, :, 2] = shift
        grid = F.affine_grid(theta.to(x.device), list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        y = F.grid_sample(y, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # photometric changes on the two image channels only
    gain = (1 + (u(b) * 2 - 1) * cfg.get("gain", 0.15)).to(x.device).view(b, 1, 1, 1)
    bias = ((u(b) * 2 - 1) * cfg.get("bias", 0.15)).to(x.device).view(b, 1, 1, 1)
    img = x[:, :2] * gain + bias
    if cfg.get("p_blur", 0) > 0 and u(1) < cfg["p_blur"]:
        img = _gaussian_blur(img, float(0.5 + u(1) * cfg.get("max_blur", 1.0)))
    if cfg.get("p_noise", 0) > 0 and u(1) < cfg["p_noise"]:
        img = img + torch.randn(img.shape, device=x.device) * float(u(1) * cfg.get("max_noise", 0.08))
    if cfg.get("p_haze", 0) > 0 and u(1) < cfg["p_haze"]:  # smooth additive brightness field on intensity
        field = torch.randn(b, 1, 4, 4, device=x.device) * cfg.get("haze_amp", 0.3)
        img[:, :1] = img[:, :1] + F.interpolate(field, size=x.shape[-2:], mode="bicubic", align_corners=False)
    x = torch.cat([img, x[:, 2:]], dim=1)
    return x, y.clamp(0, 1)
