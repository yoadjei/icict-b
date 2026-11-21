"""Frozen Prithvi-EO-1.0-100M and ViT-Base embeddings of the Sentinel-2 patches.

Preprocessing: masked pixels are filled with the band median; Prithvi receives the six HLS
order bands normalised with the model's published statistics; ViT receives RGB clipped to
[0, 1] with ImageNet normalisation; both are resized bilinearly from 64 x 64 to 224 x 224
and the CLS token (768 values) is kept.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cropshift.earth_engine import S2_BANDS

TARGET = 224
PRITHVI_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]
PRITHVI_MEAN = np.array([775.23, 1080.99, 1228.59, 2497.20, 2204.21, 1610.83], dtype=np.float32) / 10000.0
PRITHVI_STD = np.array([1281.53, 1270.03, 1399.48, 1368.34, 1291.68, 1154.51], dtype=np.float32) / 10000.0
PRITHVI_ARGS = dict(img_size=224, in_chans=6, num_frames=1, patch_size=16, tubelet_size=1, embed_dim=768,
                    depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16)
VIT_NAME = "google/vit-base-patch16-224"
VIT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
VIT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def fill_masked(bands: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Channels first float32 patch with invalid pixels set to the band median of valid pixels."""
    out = np.moveaxis(bands, -1, 0).astype(np.float32).copy()
    mask = valid.astype(bool)
    for c in range(out.shape[0]):
        fill = float(np.median(out[c][mask])) if mask.any() else 0.0
        out[c][~mask] = fill
    return out


def _resize(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x).unsqueeze(0)
    return F.interpolate(t, size=(TARGET, TARGET), mode="bilinear", align_corners=False).squeeze(0)


def prithvi_input(patch: np.ndarray) -> torch.Tensor:
    idx = [S2_BANDS.index(b) for b in PRITHVI_BANDS]
    x = (patch[idx] - PRITHVI_MEAN[:, None, None]) / (PRITHVI_STD[:, None, None] + 1e-8)
    return _resize(x)


def vit_input(patch: np.ndarray) -> torch.Tensor:
    idx = [S2_BANDS.index(b) for b in ("B4", "B3", "B2")]
    x = (np.clip(patch[idx], 0, 1) - VIT_MEAN[:, None, None]) / (VIT_STD[:, None, None] + 1e-8)
    return _resize(x)


def load_prithvi(weights: Path):
    from cropshift.prithvi_mae import PrithviMAE
    model = PrithviMAE(**PRITHVI_ARGS)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if "pos_embed" not in k}  # recomputed from the grid
    model.load_state_dict(state, strict=False)
    return model.eval()


def load_vit():
    from transformers import ViTModel
    return ViTModel.from_pretrained(VIT_NAME).eval()


@torch.no_grad()
def prithvi_cls(model, batch: torch.Tensor) -> np.ndarray:
    features = model.forward_features(batch.unsqueeze(2))  # (B, 6, 1, 224, 224)
    return features[-1][:, 0, :].float().numpy()


@torch.no_grad()
def vit_cls(model, batch: torch.Tensor) -> np.ndarray:
    return model(pixel_values=batch).last_hidden_state[:, 0, :].float().numpy()
