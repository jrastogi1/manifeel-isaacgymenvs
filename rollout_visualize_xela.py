# Rollout visualization + Tactile-JEPA embedding extraction for AllegroKukaXela.
#
# Runs a scripted contact rollout (same action scheme as smoke_test_xela.py),
# records every control step's taxel observation, and then:
#   * encodes the 60 Hz taxel stream with the Tactile-JEPA adapter
#     (resampled to the training's 100 Hz, 10-frame windows);
#   * saves embeddings + raw stream to an .npz;
#   * renders an overlay video (env viewer frame + taxel panel + embedding
#     panel) using the viewer's `record_frames` PNG path - this deliberately
#     avoids the `capture_video`/SmartDisplay route, which needs Xvfb and is
#     not installed on this machine.
#
# The viewer camera tracks the palm (camera = palm + --cam_offset), because the
# inherited AllegroKuka camera presets target (6, 5, 0) ~7 m away from this
# task's workspace and leave the robot tiny in the corner of the frame.
#
# Run from the manifeel-isaacgymenvs directory:
#   python rollout_visualize_xela.py                 # viewer + overlay video
#   python rollout_visualize_xela.py --no_frames     # headless, panels-only video
#   python rollout_visualize_xela.py --steps 300 --save_patch
#
# Outputs (default ./tactile_jepa_outputs/<timestamp>/):
#   rollout.npz      taxels [T,368,3], actions, rewards, pooled [W,192], ...
#   overlay.mp4      side-by-side visualization (unless --no_video)
#   viewer_frames/   raw env PNGs (when frames are enabled)

import argparse
import glob
import os
import re
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isaacgym  # must be imported before torch per Isaac Gym requirement
import torch

from isaacgym import gymapi

import isaacgymenvs
from isaacgymenvs.tasks.allegro_kuka.xela_sensor_config import (
    NUM_XELA_TAXELS,
    TAXEL_DIM,
    XELA_FLATTEN_ORDER,
)
from tactile_jepa_adapter import (
    EMBED_DIM,
    NUM_TAXELS,
    TARGET_FREQ_HZ,
    WINDOW_FRAMES,
    TactileJEPAEncoder,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300, help="control steps to run")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--sim_device", type=str, default="cuda:0")
    p.add_argument("--rl_device", type=str, default="cuda:0")
    p.add_argument("--graphics_device_id", type=int, default=0)
    p.add_argument("--no_frames", action="store_true",
                   help="headless rollout (no env RGB; panels-only video)")
    p.add_argument("--no_video", action="store_true",
                   help="skip overlay video rendering (still saves the .npz)")
    p.add_argument("--save_patch", action="store_true",
                   help="also save per-window patch tokens [W,368,192] (~30 MB)")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="JEPA device cuda|cpu (default: cuda if available)")
    p.add_argument("--cam_offset", type=float, nargs=3, default=[0.55, -0.7, 0.35],
                   metavar=("DX", "DY", "DZ"),
                   help="camera offset relative to the aim target (default: "
                        "0.55 -0.7 0.35, ~1 m 3/4 front-left view)")
    p.add_argument("--static_workspace_camera", action="store_true",
                   help="aim once at the fixed workspace centre (0, 0.3, 0.55) "
                        "instead of tracking the palm")
    return p.parse_args()


def patch_boundaries():
    """(taxel_index, patch_name) of each patch start, for drawing separators."""
    starts, acc = [], 0
    for name, count in XELA_FLATTEN_ORDER.items():
        starts.append((acc, name))
        acc += count
    return starts


def make_panel(step, t_s, taxel_now, mag_history, axis_history, pooled, win_idx,
               n_steps, contact_max, patch_starts):
    """Render one visualization panel -> HxWx3 uint8 RGB."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), dpi=90,
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    mag = np.linalg.norm(taxel_now, axis=-1)               # [368]

    ax = axes[0]
    ax.bar(np.arange(NUM_TAXELS), mag, width=1.0, color="#1f77b4")
    for i, (start, _) in enumerate(patch_starts[1:], start=1):
        ax.axvline(start - 0.5, color="0.85", lw=0.6)
    ax.set_xlim(0, NUM_TAXELS)
    ax.set_ylim(0, max(0.05, mag_history.max() * 1.05, contact_max * 1.05))
    ax.set_title("taxel |f| per taxel (patch-separated)", fontsize=8)
    ax.set_xlabel("taxel index", fontsize=7)
    ax.tick_params(labelsize=6)

    ax = axes[1]
    tt = np.arange(axis_history.shape[0])
    for c, (lbl, col) in enumerate(zip(("X shear", "Y shear", "Z normal"),
                                       ("#d62728", "#2ca02c", "#9467bd"))):
        ax.plot(tt, axis_history[: len(axis_history), c], color=col, label=lbl, lw=1.0)
    ax.set_xlim(0, max(n_steps, 1))
    ax.set_title("axis totals |sum| over control steps", fontsize=8)
    ax.set_xlabel("control step", fontsize=7)
    ax.legend(fontsize=6, loc="upper left")
    ax.tick_params(labelsize=6)

    ax = axes[2]
    show = pooled[: max(win_idx + 1, 1)]
    ax.imshow(show.T, aspect="auto", origin="lower", cmap="viridis",
              vmin=pooled.min(), vmax=pooled.max())
    ax.set_title("pooled embedding [W, %d] (t up to %.2fs)" % (EMBED_DIM, t_s),
                 fontsize=8)
    ax.set_xlabel("window", fontsize=7)
    ax.set_ylabel("dim", fontsize=7)
    ax.tick_params(labelsize=6)

    fig.suptitle("step %d  t=%.2fs  max|f|=%.3f N  contact=%.3f"
                 % (step, t_s, float(np.abs(taxel_now).max()), contact_max),
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def main():
    args = parse_args()
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out_dir or os.path.join("tactile_jepa_outputs", stamp)
    frames_wanted = not args.no_frames
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------- env --
    env = isaacgymenvs.make(
        seed=0,
        task="AllegroKukaXela",
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=args.graphics_device_id if frames_wanted else -1,
        headless=not frames_wanted,
        multi_gpu=False,
        virtual_screen_capture=False,   # SmartDisplay needs Xvfb (not installed)
        force_render=True,              # vec_task.step() then calls render()
        cfg=None,
    )
    proprio = env.base_full_state_size
    assert env.num_observations == proprio + NUM_XELA_TAXELS * TAXEL_DIM
    src_freq = 1.0 / (env.dt * env.control_freq_inv)

    frames_dir = os.path.join(out_dir, "viewer_frames")
    if frames_wanted:
        env.record_frames = True          # render() writes frame_<n>.png per step
        env.record_frames_dir = frames_dir
    print("[rollout] num_envs=%d  control rate=%.1f Hz  frames=%s"
          % (args.num_envs, src_freq, frames_wanted))

    # -------------------------------------------------------------- camera --
    # VecTask/AllegroKuka point the viewer camera at (6, 5, 0) - about 7 m away
    # from this task's workspace (robot base (0, 0.8, 0), object (0, 0, 0.63)) -
    # which leaves the robot tiny and off-centre in the frame. Instead we aim at
    # the palm: camera = palm + --cam_offset (re-aimed every step, so the hand
    # stays centred even as it moves).
    cam_offset = np.asarray(args.cam_offset, dtype=np.float64)
    workspace_center = np.array([0.0, 0.3, 0.55], dtype=np.float64)

    def aim_camera():
        """Aim the viewer at the palm (or the workspace centre in static mode)."""
        if env.viewer is None:
            return
        if args.static_workspace_camera:
            tgt = workspace_center
        else:
            tgt = env.palm_center_pos[0].detach().cpu().numpy().astype(np.float64)
        pos = tgt + cam_offset
        env.gym.viewer_camera_look_at(
            env.viewer, None,
            gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2])),
            gymapi.Vec3(float(tgt[0]), float(tgt[1]), float(tgt[2])),
        )

    # ------------------------------------------------------------- rollout --
    obs_dict = env.reset()
    obs = obs_dict["obs"]
    # reset() does not compute observations; do one pass so palm_center_pos
    # exists before the first render, then aim the camera.
    env.compute_observations()
    aim_camera()
    print("[camera] aiming at %s with offset %s"
          % ("workspace centre" if args.static_workspace_camera else "palm",
             list(cam_offset)))
    actions = torch.zeros((args.num_envs, env.num_actions),
                          device=args.rl_device, dtype=torch.float)

    taxel_seq, action_log, reward_log, contact_max = [], [], [], 0.0
    for step in range(args.steps):
        # re-aim the viewer before stepping (render() happens inside step);
        # one-step lag on the palm is irrelevant for framing.
        if not args.static_workspace_camera:
            aim_camera()
        # scripted contact grasp (identical scheme to smoke_test_xela.py)
        actions.zero_()
        actions[:, 7:] = min(1.0, step / 60.0)   # close the fingers
        actions[:, 0] = 0.2                      # small arm approach
        obs_dict, rew, reset, _ = env.step(actions)
        obs = obs_dict["obs"]

        taxels = obs[:, proprio:].reshape(
            args.num_envs, NUM_XELA_TAXELS, TAXEL_DIM)
        t0 = taxels[0].detach().cpu().numpy().astype(np.float32)
        taxel_seq.append(t0)
        action_log.append(actions[0].detach().cpu().numpy().astype(np.float32))
        reward_log.append(float(rew[0].item()))
        contact_max = max(contact_max, float(np.abs(t0).max()))
        if step % 60 == 0:
            print("  step %3d | max|taxel| %.4f | rew %.3f | resets %d"
                  % (step, float(np.abs(t0).max()), float(rew[0].item()),
                     int(reset.sum().item())))

    taxel_seq = np.stack(taxel_seq)                       # [T, 368, 3]
    action_log = np.stack(action_log)
    reward_log = np.asarray(reward_log)
    print("[rollout] done: T=%d, contact_max=%.4f" % (len(taxel_seq), contact_max))
    assert contact_max > 1e-3, "taxels never activated - no contact in rollout"

    # ------------------------------------------------- Tactile-JEPA encode --
    adapter = TactileJEPAEncoder(device=args.device)
    dense = adapter.encode_rollout(taxel_seq, src_freq_hz=src_freq, stride_frames=1)
    print("[jepa] dense windows: %s @ %.1f->%.0f Hz"
          % (str(dense["pooled"].shape), src_freq, TARGET_FREQ_HZ))
    save = {
        "taxels": taxel_seq,
        "actions": action_log,
        "rewards": reward_log,
        "src_freq_hz": src_freq,
        "pooled": dense["pooled"],
        "window_times": dense["times"],
        "resampled": dense["resampled"],
        "contact_max": contact_max,
    }
    if args.save_patch:
        sparse = adapter.encode_rollout(taxel_seq, src_freq_hz=src_freq,
                                        stride_frames=WINDOW_FRAMES,
                                        return_patch=True)
        save["patch"] = sparse["patch"]
        save["patch_window_times"] = sparse["times"]

    npz_path = os.path.join(out_dir, "rollout.npz")
    np.savez_compressed(npz_path, **save)
    print("[save] %s (%.1f MB)"
          % (npz_path, os.path.getsize(npz_path) / 1e6))

    # -------------------------------------------------------------- video --
    if args.no_video:
        print("\nDONE (video skipped): %s" % out_dir)
        return

    import cv2

    panel_starts = patch_boundaries()
    mag_history = np.zeros((0, NUM_TAXELS), dtype=np.float32)
    pooled = dense["pooled"]
    # [T, 3]: per-step sum of |force| over taxels, one column per axis
    axis_history = np.stack(
        [np.abs(taxel_seq[:, :, c]).sum(axis=1) for c in range(TAXEL_DIM)], axis=1
    )

    # collect viewer PNGs written during rollout (frame_<control_steps>.png)
    pngs = []
    if frames_wanted and os.path.isdir(frames_dir):
        def _num(path):
            m = re.search(r"frame_(\d+)\.png$", path)
            return int(m.group(1)) if m else -1
        pngs = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")), key=_num)
    print("[video] %d viewer frames, %d steps" % (len(pngs), args.steps))

    panel0 = make_panel(0, 0.0, taxel_seq[0],
                        np.linalg.norm(taxel_seq[:1], axis=-1),
                        axis_history[:1], pooled, 0, args.steps, contact_max,
                        panel_starts)
    ph, pw = panel0.shape[:2]
    writer = None
    video_path = os.path.join(out_dir, "overlay.mp4")

    for step in range(args.steps):
        t_s = step / src_freq
        # nearest dense JEPA window (windows live on the 100 Hz clock)
        win_idx = min(int(round(t_s * TARGET_FREQ_HZ)), pooled.shape[0] - 1)
        mag_now = np.linalg.norm(taxel_seq[step], axis=-1)      # [368]
        mag_history = np.append(mag_history, mag_now[None], axis=0)
        panel = make_panel(step, t_s, taxel_seq[step], mag_history,
                           axis_history[: step + 1], pooled, win_idx,
                           args.steps, contact_max, panel_starts)

        rgb = panel
        if pngs:
            png_path = pngs[min(step, len(pngs) - 1)]
            img = cv2.cvtColor(cv2.imread(png_path), cv2.COLOR_BGR2RGB)
            scale = ph / img.shape[0]
            img = cv2.resize(img, (max(int(img.shape[1] * scale), 1), ph))
            sep = np.full((ph, 4, 3), 255, dtype=np.uint8)
            rgb = np.concatenate([img, sep, panel], axis=1)

        if writer is None:
            h, w = rgb.shape[:2]
            fps = max(src_freq, 1.0)
            writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                video_path = video_path.replace(".mp4", ".avi")
                writer = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    if writer is not None:
        writer.release()
    print("[video] wrote %s" % video_path)
    print("\nDONE: %s\n  windows=%s pooled -> %s"
          % (out_dir, pooled.shape, npz_path))


if __name__ == "__main__":
    main()
