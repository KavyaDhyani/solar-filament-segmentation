"""Segmentation network and test-time augmentation."""

from __future__ import annotations

import torch

from .augment import d4, d4_inverse


def build_model(arch: str = "unet", encoder: str = "tu-tf_efficientnetv2_s", weights: str | None = "imagenet") -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    cls = getattr(smp, {"unet": "Unet", "unetpp": "UnetPlusPlus", "fpn": "FPN", "upernet": "UPerNet"}[arch])
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1)


def amp_dtype() -> torch.dtype:
    """bfloat16 where the GPU supports it natively (Ampere+), else float16 (e.g. T4)."""
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


@torch.no_grad()
def forward_safe(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Mixed-precision forward; samples whose fp16 activations overflow (non-finite logits) are redone in fp32."""
    if x.device.type != "cuda":  # CPU inference: plain float32
        return model(x).float()
    with torch.autocast("cuda", dtype=amp_dtype()):
        out = model(x).float()
    bad = ~torch.isfinite(out).flatten(1).all(1)
    if bad.any():
        out[bad] = model(x[bad].float()).float()
    return out


@torch.no_grad()
def predict_tta(model: torch.nn.Module, x: torch.Tensor, views: int = 8) -> torch.Tensor:
    """Mean sigmoid probability over the first ``views`` dihedral transforms. x: (B, 3, S, S) -> (B, S, S)."""
    acc = torch.zeros(x.shape[0], 1, *x.shape[-2:], device=x.device)
    ops = [(k, f) for f in (False, True) for k in range(4)][:views]
    for k, flip in ops:
        xt = d4(x, k, flip).contiguous(memory_format=torch.channels_last)
        acc += d4_inverse(forward_safe(model, xt).sigmoid(), k, flip)
    return acc[:, 0] / len(ops)
