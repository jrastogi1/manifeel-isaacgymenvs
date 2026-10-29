# Smoke test for the Tactile-JEPA inference adapter (no Isaac Gym needed).
#
# Verifies, against the real HF checkpoint (pretrain/encoder.safetensors):
#   * strict state-dict load (107 keys, depth=8, embed_dim=192);
#   * input contract  [B, 10, 368, 3] and output shapes ([B,368,192] / [B,192]);
#   * outputs are finite, deterministic, and react to the input;
#   * rollout path: [T,368,3] @60 Hz -> resample to 100 Hz -> sliding windows.
#
# Run from the manifeel-isaacgymenvs directory:
#   python smoke_test_tactile_jepa.py
# Options: --device cpu, --steps N, --save_patch (exercise patch-token path)

import argparse

import numpy as np
import torch

from tactile_jepa_adapter import (
    EMBED_DIM,
    NUM_TAXELS,
    TAXEL_DIM,
    TARGET_FREQ_HZ,
    WINDOW_FRAMES,
    TactileJEPAEncoder,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default=None,
                   help="cuda|cpu (default: cuda if available)")
    p.add_argument("--steps", type=int, default=600,
                   help="rollout length in control steps (default 600 ~ 10 s @60Hz)")
    p.add_argument("--src_freq", type=float, default=60.0,
                   help="control-step rate of the stream being encoded")
    p.add_argument("--save_patch", action="store_true",
                   help="also run the (W,368,192) patch-token path")
    return p.parse_args()


def main():
    args = parse_args()
    adapter = TactileJEPAEncoder(device=args.device)

    # ---- printed checkpoint facts ---------------------------------------
    print("xela_mean:", adapter.xela_mean, " xela_std:", adapter.xela_std)
    n_params = sum(p.numel() for p in adapter.model.parameters())
    print("parameters: {:,}".format(n_params))

    # ---- 1. shape + finiteness on synthetic windows ---------------------
    zeros = torch.zeros(4, WINDOW_FRAMES, NUM_TAXELS, TAXEL_DIM)
    out0 = adapter.encode(zeros)
    assert tuple(out0["patch"].shape) == (4, NUM_TAXELS, EMBED_DIM), out0["patch"].shape
    assert tuple(out0["pooled"].shape) == (4, EMBED_DIM), out0["pooled"].shape
    assert torch.isfinite(out0["patch"]).all(), "non-finite patch tokens on zeros"
    assert torch.isfinite(out0["pooled"]).all(), "non-finite pooled on zeros"
    print("[1] zero-window shapes + finiteness OK")

    # ---- 2. determinism --------------------------------------------------
    out0b = adapter.encode(zeros)
    assert torch.equal(out0["pooled"], out0b["pooled"]), "encoder is not deterministic"
    print("[2] determinism OK")

    # ---- 3. reacts to input: structured signal != zeros ------------------
    g = torch.Generator().manual_seed(0)
    rand = torch.randn(4, WINDOW_FRAMES, NUM_TAXELS, TAXEL_DIM, generator=g) * 0.5
    outr = adapter.encode(rand)
    diff = (out0["pooled"] - outr["pooled"]).abs().max().item()
    assert diff > 1e-4, "embeddings do not react to input (max diff %g)" % diff
    intra = (outr["pooled"][0] - outr["pooled"][1]).abs().max().item()
    assert intra > 1e-6, "identical embeddings for different windows"
    assert torch.isfinite(outr["patch"]).all()
    print("[3] input sensitivity OK (max |zeros-rand| pooled = %.4f)" % diff)

    # ---- 4. wrong shapes are rejected ------------------------------------
    try:
        adapter.encode(torch.zeros(4, WINDOW_FRAMES - 1, NUM_TAXELS, TAXEL_DIM))
        raise AssertionError("bad window length was not rejected")
    except ValueError:
        pass
    print("[4] shape validation OK")

    # ---- 5. rollout pipeline @60Hz -> 100Hz -> windows --------------------
    t = args.steps
    # synthetic contact-like burst: taxels activate between 30%-70% of rollout
    seq = np.zeros((t, NUM_TAXELS, TAXEL_DIM), dtype=np.float32)
    start, stop = int(0.3 * t), int(0.7 * t)
    ramp = np.linspace(0.0, 2.0, stop - start).astype(np.float32)
    seq[start:stop, :, 2] = ramp[:, None]            # normal force (Z), all taxels
    seq[start:stop, :120, 0] = 0.3 * ramp[:, None]   # shear on first four patches

    res = adapter.encode_rollout(
        seq, src_freq_hz=args.src_freq, stride_frames=WINDOW_FRAMES,
        return_patch=args.save_patch,
    )
    n_res = int(round(t / args.src_freq * TARGET_FREQ_HZ))
    n_win = max(0, (n_res - WINDOW_FRAMES) // WINDOW_FRAMES + 1)
    assert tuple(res["pooled"].shape) == (n_win, EMBED_DIM), res["pooled"].shape
    assert np.isfinite(res["pooled"]).all(), "non-finite rollout embeddings"
    assert res["times"].shape == (n_win,)
    assert res["times"][0] == 0.0 and res["times"][-1] <= t / args.src_freq + 1e-6
    assert tuple(res["resampled"].shape) == (n_res, NUM_TAXELS, TAXEL_DIM)
    if args.save_patch:
        assert tuple(res["patch"].shape) == (n_win, NUM_TAXELS, EMBED_DIM)
    # embeddings must differ between free space and the contact burst
    dur = t / args.src_freq
    free = [i for i, ti in enumerate(res["times"]) if ti < 0.25 * dur]
    contact = [i for i, ti in enumerate(res["times"]) if 0.4 * dur < ti < 0.6 * dur]
    if free and contact:
        sep = np.linalg.norm(
            res["pooled"][free].mean(0) - res["pooled"][contact].mean(0)
        )
        print("    pooled ||free - contact|| = %.4f" % sep)
        assert sep > 1e-4, "embedding does not distinguish contact from free space"
    print("[5] rollout pipeline OK: %d steps -> %d windows (60->100 Hz resample)"
          % (t, res["pooled"].shape[0]))

    # ---- 6. dense stride works too ----------------------------------------
    dense = adapter.encode_rollout(seq[:200], src_freq_hz=args.src_freq, stride_frames=1)
    expect = int(round(200 / args.src_freq * TARGET_FREQ_HZ)) - WINDOW_FRAMES + 1
    assert dense["pooled"].shape[0] == expect, dense["pooled"].shape
    print("[6] stride=1 dense windows OK (%d windows)" % dense["pooled"].shape[0])

    print("\nSMOKE TEST PASSED: Tactile-JEPA adapter ready "
          "(in [B,%d,%d,%d] -> pooled [B,%d])"
          % (WINDOW_FRAMES, NUM_TAXELS, TAXEL_DIM, EMBED_DIM))


if __name__ == "__main__":
    main()
