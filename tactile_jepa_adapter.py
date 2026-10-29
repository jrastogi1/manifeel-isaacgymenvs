#!/usr/bin/env python3
"""Tactile-JEPA encoder adapter for the AllegroKukaXela tactile stream.

Bridges simulated XELA taxel observations (368 taxels x 3 axes) to the
Tactile-JEPA encoder pretrained on real XELA uSkin data:

  * weights:  HF  Tactile-JEPA/tactile-jepa-xela   (pretrain/encoder.safetensors)
  * code:     GH   E-Kovtun/tactile                (cloned locally; this module
              puts DEFAULT_TACTILE_REPO on sys.path itself)

Contract verified against upstream source (`tactile_ssl/model/signal_transformer.py`,
`xela_transformer.py`) and the shipped checkpoint (107 keys, strict load):

  * input windows : float32 [B, T=10, N=368, C=3]  (time-major, `b t n c`)
  * training data : 0.1 s windows at 100 Hz (window_time=0.1,
    interpolating_freq=100, window_overlap=0); taxel forces in Newtons
  * normalization : per-axis (xela_mean, xela_std) are *persistent buffers*,
    loaded from the checkpoint itself - no external stats needed
  * output        : forward_features() dict; exposed here as
                      "patch"  = x_norm_patchtokens [B, 368, 192]
                      "pooled" = mean over the 368 taxel tokens  [B, 192]

Python 3.8 compatible (matches the manifeel venv).

Example
-------
    adapter = TactileJEPAEncoder()                       # downloads HF weights once
    out = adapter.encode(windows)                        # [B, 10, 368, 3] -> dict
    out = adapter.encode_rollout(seq, src_freq_hz=60.0)  # [T, 368, 3] stream
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_TACTILE_REPO = "/home/jrastogi/Documents/cs_projects/manifeel/tactile"
DEFAULT_HF_REPO = "Tactile-JEPA/tactile-jepa-xela"
DEFAULT_HF_SUBPATH = "pretrain/encoder.safetensors"

NUM_TAXELS = 368
TAXEL_DIM = 3
WINDOW_FRAMES = 10           # sequence_length == time_chunk_size (100 Hz * 0.1 s)
TARGET_FREQ_HZ = 100.0       # interpolating_freq used in Tactile-JEPA training
EMBED_DIM = 192

# Exact `algorithm.encoder` block from HF pretrain/training_config.yaml.
# `xela_tiny` factory defaults give depth=8 / embed_dim=192 / num_heads=3, which
# matches the checkpoint (blocks.0..7). `normalization` was null in training;
# the real stats ship inside the checkpoint's persistent buffers.
ENCODER_KWARGS = dict(
    in_dim=NUM_TAXELS,
    in_chans=TAXEL_DIM,
    sequence_length=WINDOW_FRAMES,
    time_chunk_size=WINDOW_FRAMES,
    temporal_patch_padding="auto",
    num_register_tokens=0,
    pos_embed_fn="learned",
    with_masktoken=False,
    drop_path_rate=0.3,      # inactive in eval(); kept for exact parity
    drop_path_uniform=True,
    normalization=None,
    causal=False,
)

_CKPT_PREFIXES = ("", "encoder.", "target_encoder.", "module.", "model.")


def _import_xela_tiny(tactile_repo: str = DEFAULT_TACTILE_REPO):
    """Put the Tactile repo on sys.path and import the encoder factory."""
    if not os.path.isdir(os.path.join(tactile_repo, "tactile_ssl")):
        raise FileNotFoundError(
            "Tactile repo not found at {!r}. Clone it first:\n"
            "  git clone https://github.com/E-Kovtun/tactile {}".format(
                tactile_repo, tactile_repo
            )
        )
    if tactile_repo not in sys.path:
        sys.path.insert(0, tactile_repo)
    from tactile_ssl.model.xela_transformer import xela_tiny  # noqa: WPS433

    return xela_tiny


def _resolve_ckpt(ckpt_path: Optional[str]) -> str:
    if ckpt_path is not None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError("checkpoint not found: {!r}".format(ckpt_path))
        return ckpt_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DEFAULT_HF_REPO, DEFAULT_HF_SUBPATH)


def _load_state_dict(model: torch.nn.Module, path: str) -> None:
    """Load `.safetensors` or `.pth/.ckpt` with strict=True (prefix-tolerant)."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        for outer in ("state_dict", "model", "encoder"):  # trainer ckpt nesting
            if isinstance(sd, dict) and outer in sd and isinstance(sd[outer], dict):
                sd = sd[outer]
                break

    errors = []
    for prefix in _CKPT_PREFIXES:
        if prefix and not all(k.startswith(prefix) for k in sd):
            continue
        candidate = sd if not prefix else {k[len(prefix):]: v for k, v in sd.items()}
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except RuntimeError as exc:  # try next prefix / report below
            errors.append(str(exc))
    raise RuntimeError(
        "strict state-dict load failed for {!r}:\n{}".format(path, "\n---\n".join(errors))
    )


class TactileJEPAEncoder:
    """Loads the pretrained XELA encoder and maps taxel rollouts to embeddings."""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
        tactile_repo: str = DEFAULT_TACTILE_REPO,
        model_size: str = "tiny",
    ) -> None:
        xela_tiny = _import_xela_tiny(tactile_repo)
        if model_size != "tiny":
            raise ValueError("only the 'tiny' checkpoint is published, got %r" % model_size)

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = xela_tiny(**ENCODER_KWARGS)
        self.model.to(self.device)
        self.model.eval()

        path = _resolve_ckpt(ckpt_path)
        self.ckpt_path = path
        _load_state_dict(self.model, path)

        # persistent buffers - loaded from the checkpoint itself, shape (3,)
        self.xela_mean = self.model.xela_mean.detach().cpu().numpy()
        self.xela_std = self.model.xela_std.detach().cpu().numpy()
        marker = int(self.model._temporal_patch_version.item())
        print(
            "[TactileJEPA] loaded {} (device={}, temporal_patch_version={}, "
            "xela_mean={}, xela_std={})".format(
                path, self.device, marker,
                np.round(self.xela_mean, 4).tolist(),
                np.round(self.xela_std, 4).tolist(),
            )
        )

    # ------------------------------------------------------------------ #
    # core encode
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(
        self, windows: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Encode [B, 10, 368, 3] windows -> {"patch": [B,368,192], "pooled": [B,192]}."""
        x = torch.as_tensor(windows, dtype=torch.float32)
        if x.ndim != 4 or x.shape[1:] != (WINDOW_FRAMES, NUM_TAXELS, TAXEL_DIM):
            raise ValueError(
                "expected windows [B, {}, {}, {}], got {}".format(
                    WINDOW_FRAMES, NUM_TAXELS, TAXEL_DIM, tuple(x.shape)
                )
            )
        x = x.to(self.device)
        out = self.model.forward_features(x)
        patch = out["x_norm_patchtokens"]                 # [B, 368, 192]
        pooled = patch.mean(dim=1)                        # [B, 192]
        return {"patch": patch, "pooled": pooled, "regtokens": out["x_norm_regtokens"]}

    # ------------------------------------------------------------------ #
    # rollout stream -> windows
    # ------------------------------------------------------------------ #
    @staticmethod
    def resample_sequence(
        seq: Union[np.ndarray, torch.Tensor], src_freq_hz: float
    ) -> torch.Tensor:
        """Linearly resample a [T, 368, 3] stream to the training rate (100 Hz)."""
        x = torch.as_tensor(seq, dtype=torch.float32)
        if x.ndim != 3 or x.shape[1:] != (NUM_TAXELS, TAXEL_DIM):
            raise ValueError(
                "expected sequence [T, {}, {}], got {}".format(
                    NUM_TAXELS, TAXEL_DIM, tuple(x.shape)
                )
            )
        t = x.shape[0]
        if src_freq_hz <= 0:
            raise ValueError("src_freq_hz must be positive, got {}".format(src_freq_hz))
        if abs(src_freq_hz - TARGET_FREQ_HZ) < 1e-6:
            return x
        t_target = max(int(round(t / src_freq_hz * TARGET_FREQ_HZ)), WINDOW_FRAMES)
        # [T, N, C] -> [1, N*C, T] -> linear interp over time -> back
        flat = x.permute(1, 2, 0).reshape(1, NUM_TAXELS * TAXEL_DIM, t)
        res = F.interpolate(flat, size=t_target, mode="linear", align_corners=True)
        return res.reshape(NUM_TAXELS, TAXEL_DIM, t_target).permute(2, 0, 1)

    @staticmethod
    def sliding_windows(
        seq: torch.Tensor, stride_frames: int = WINDOW_FRAMES
    ) -> torch.Tensor:
        """[T', 368, 3] (at 100 Hz) -> [W, 10, 368, 3]; edge-pads if shorter."""
        if stride_frames < 1:
            raise ValueError("stride_frames must be >= 1")
        if seq.shape[0] < WINDOW_FRAMES:  # edge-pad up to one full window
            pad = seq[-1:].repeat(WINDOW_FRAMES - seq.shape[0], 1, 1)
            seq = torch.cat([seq, pad], dim=0)
        windows = seq.unfold(0, WINDOW_FRAMES, stride_frames)  # [W, N, C, 10]
        return windows.permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def encode_rollout(
        self,
        seq: Union[np.ndarray, torch.Tensor],
        src_freq_hz: float,
        stride_frames: int = WINDOW_FRAMES,
        batch_size: int = 256,
        return_patch: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Turn a raw taxel stream into per-window embeddings.

        Parameters
        ----------
        seq : [T, 368, 3] taxel forces sampled at ``src_freq_hz``
              (AllegroKukaXela: 1 / (env.dt * env.control_freq_inv) ~= 60 Hz).
        stride_frames : hop between windows **in resampled 100 Hz frames**.
              WINDOW_FRAMES (10) = non-overlapping, matching training
              (window_overlap=0); use 1 for a dense per-frame trajectory.

        Returns
        -------
        numpy-backed dict: "pooled" [W, 192], optional "patch" [W, 368, 192],
        "times" [W] window-start seconds (original clock), "resampled" [T', 368, 3].
        """
        resampled = self.resample_sequence(seq, src_freq_hz)
        windows = self.sliding_windows(resampled, stride_frames=stride_frames)

        pooled_chunks, patch_chunks = [], []
        for i in range(0, windows.shape[0], batch_size):
            out = self.encode(windows[i: i + batch_size])
            pooled_chunks.append(out["pooled"])
            if return_patch:
                patch_chunks.append(out["patch"])
        pooled = torch.cat(pooled_chunks, dim=0)

        result = {
            "pooled": pooled.cpu().numpy(),
            "times": (torch.arange(windows.shape[0]) * stride_frames
                      / TARGET_FREQ_HZ).numpy(),
            "resampled": resampled.numpy(),
        }
        if return_patch:
            result["patch"] = torch.cat(patch_chunks, dim=0).cpu().numpy()
        return result
