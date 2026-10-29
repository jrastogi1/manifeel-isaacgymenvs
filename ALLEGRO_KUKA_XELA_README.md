# AllegroKukaXela — 16-DOF Allegro hand + XELA tactile sensors on a 7-DOF KUKA arm

An Isaac Gym RL environment that mounts **XELA uSkin electronic-skin patches** on a
16-DOF Wonik Allegro hand carried by a 7-DOF KUKA iiwa7 arm, matching the sensor
layout used by the Tactile-JEPA reference setup
(<https://huggingface.co/Tactile-JEPA/tactile-jepa-xela>).

The observation concatenates the **proprioception state** with the flattened
**taxel** readings (368 taxels × 3 axes), so the agent sees both robot state and
distributed tactile feedback. The task is in-hand object **reorientation**
(reward/reset logic inherited from `AllegroKukaReorientation`).

## Sensor layout (matches Tactile-JEPA `XELA_FLATTEN_ORDER`)

18 XELA uSkin patches -> **368 taxels**, each reporting **3-axis touch**
(X = shear, Y = shear, Z = normal):

| Region | Patches | Taxels/patch | Subtotal |
|---|---|---|---|
| 4 fingertips (curved `aftc`) | `{0,1,2,3}aftc_palm_link` | 30 | 120 |
| 11 finger phalanges (flat `4x4`) | `link_{...}_4x4_palm_link` | 16 | 176 |
| 3 palm patches (`4x6`) | `ahr_palm_{1,2,3}_4x6_palm_link` | 24 | 72 |
| **Total** | **18** | | **368** |

The single source of truth for the ordering, hand-part grouping
(thumb / index / middle / ring / palm), per-taxel sensor-grid geometry, and the
physical connectivity graph is
`isaacgymenvs/tasks/allegro_kuka/xela_sensor_config.py` (a faithful port of the
Tactile-JEPA `tactile_ssl` XELA utilities).

## Observation

`obs = [ proprioception (117), taxels (368 × 3 = 1104) ]`  ->  `num_observations = 1221`

* **Proprioception (117):** the base `AllegroKuka` full state — 23 arm+hand DOF
  positions + 23 velocities, palm pose/velocity, object pose/velocity, fingertip
  positions relative to palm/object, keypoints, scales, progress, reward slot.
* **Taxels (1104):** the `(368, 3)` tactile tensor flattened in
  `XELA_FLATTEN_ORDER`, written right after the base state.

Actions: 23 (7 arm position-deltas + 16 finger position targets), same as
`AllegroKuka`.

## How the taxels are produced (Option A)

PhysX reports a **net contact force per rigid body**, not per taxel. Each of the
18 patch links is a distinct rigid body (`collapse_fixed_joints=False`). Every
step, for each patch we:

1. read its net contact force (world frame) from the net-contact-force tensor;
2. rotate it into the patch-local frame -> local 3-axis reading
   (shear-X, shear-Y, normal-Z);
3. estimate an effective contact point in the patch plane (from the local shear
   direction; defaults to the patch centroid);
4. distribute the local force across the patch's taxels with a Gaussian falloff
   in distance from that contact point (weights sum to 1), so taxel forces sum
   back to the patch force and nearby taxels read more than distant ones;
5. scale (`taxelForceScale`), squash with `tanh` (`taxelNormalize`), and clip.

This keeps the rigid-body count low (+18 bodies/env) while producing a spatially
varying `(368, 3)` tactile image. Tunable knobs live in
`isaacgymenvs/cfg/task/AllegroKukaXela.yaml` under the `XELA tactile` section:
`taxelForceScale`, `taxelNormalize`, `taxelClip`, `taxelSigma`,
`taxelForceThreshold`.

## Files

| File | Purpose |
|---|---|
| `assets/urdf/kuka_allegro_description/kuka_allegro_xela.urdf` | KUKA + Allegro + 18 XELA patch links (generated). |
| `isaacgymenvs/tasks/allegro_kuka/xela_sensor_config.py` | XELA layout / geometry / connectivity (Tactile-JEPA port). |
| `isaacgymenvs/tasks/allegro_kuka/gen_xela_urdf.py` | Regenerates the XELA URDF from `kuka_allegro_touch_sensor.urdf`. |
| `isaacgymenvs/tasks/allegro_kuka/allegro_kuka_xela.py` | The `AllegroKukaXela` task. |
| `isaacgymenvs/cfg/task/AllegroKukaXela.yaml` | Task config (asset, tactile knobs, sim with `contact_collection: 1`). |
| `isaacgymenvs/cfg/train/AllegroKukaXelaPPO.yaml` | rl_games PPO training config. |
| `smoke_test_xela.py` | Headless smoke test (obs shape + taxel contact behavior). |

The task is registered as `"AllegroKukaXela"` in `isaacgymenvs/tasks/__init__.py`.

## Run the smoke test

```bash
cd manifeel-isaacgymenvs
source ../env/bin/activate
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=../IsaacGym_Preview_TacSL_Package/isaacgym/python:$PYTHONPATH
python smoke_test_xela.py --num_envs 8 --steps 150
```

Expected: `obs shape (num_envs, 1221)`, taxels are exactly `0` in free space and
activate (bounded by `tanh`) once the hand closes on the object.

## Train

```bash
cd manifeel-isaacgymenvs
python isaacgymenvs/train.py task=AllegroKukaXela train=AllegroKukaXelaPPO \
    headless=True num_envs=4096
# inference rollout of a trained checkpoint (rl_games `test` mode):
python isaacgymenvs/train.py task=AllegroKukaXela test=True \
    checkpoint=runs/<name>/nn/<name>.pth num_envs=1
```

## Tactile-JEPA embeddings + rollout visualization

The tactile stream from this environment plugs directly into the pretrained
**Tactile-JEPA** encoder (<https://huggingface.co/Tactile-JEPA/tactile-jepa-xela>):

| File | Purpose |
|---|---|
| `tactile_jepa_adapter.py` | Loads the HF encoder (strict load) and maps taxel rollouts to embeddings. |
| `smoke_test_tactile_jepa.py` | Pure-model test: shapes, determinism, contact sensitivity, 60→100 Hz pipeline. |
| `rollout_visualize_xela.py` | Scripted contact rollout → embeddings `.npz` + overlay video. |

### One-time setup (already done on this machine)

```bash
git clone https://github.com/E-Kovtun/tactile ../tactile      # architecture code
pip install safetensors timm rich matplotlib lightning==2.3.3  # adapter deps
```

The clone carries a local Python-3.8 compat patch: `from __future__ import
annotations` added across `tactile_ssl/` (upstream targets ≥3.9). The HF weights
(`pretrain/encoder.safetensors`) download automatically on first use.

### Encoder contract (verified against source + checkpoint)

* input windows: `float32 [B, 10, 368, 3]` — 0.1 s at **100 Hz**, time-major;
* this env's stream is at the control rate (`1/(dt·control_freq_inv)` = **60 Hz**)
  → the adapter linearly resamples to 100 Hz, then slices sliding 10-frame
  windows (`stride_frames=10` = non-overlapping, matching training; `1` = dense);
* normalization (`xela_mean`/`xela_std`) ships **inside the checkpoint** as
  persistent buffers — loaded automatically with `strict=True` (107 keys,
  depth=8, 3.64 M params, embed dim 192);
* output: `x_norm_patchtokens [B, 368, 192]` per taxel + mean-pooled `[B, 192]`.

```bash
python smoke_test_tactile_jepa.py            # no Isaac Gym needed
python rollout_visualize_xela.py --steps 300 # env rollout + video + npz
python rollout_visualize_xela.py --no_frames # headless (panels-only video)
```

`rollout_visualize_xela.py` writes `tactile_jepa_outputs/<ts>/` containing
`rollout.npz` (`taxels [T,368,3]`, `actions`, `rewards`, `pooled [W,192]`,
`window_times`, `resampled`), `overlay.mp4` (env frame | per-taxel force bars |
axis totals | embedding heatmap) and the raw viewer PNGs.

**Camera:** the inherited AllegroKuka viewer presets aim at `(6, 5, 0)`, ~7 m
from this task's workspace — the robot looks tiny and off-centre. Both
`AllegroKukaXela` (live viewer) and `rollout_visualize_xela.py` (recordings)
now re-aim the camera at the workspace/hand; the script additionally **tracks
the palm each step** (`--cam_offset` tunes the view; `--static_workspace_camera`
locks onto the workspace centre instead).

### Video capture notes

* The overlay video uses the **viewer `record_frames` PNG path**
  (`env.record_frames = True`, requires `headless=False`) — no extra packages.
* `train.py capture_video=True` (gym `RecordVideo` + SmartDisplay) needs the
  **Xvfb** binary, which is **not installed** here — it will fail on this
  machine until `sudo apt install xvfb` is run. Prefer
  `rollout_visualize_xela.py` for recordings, or the interactive viewer
  (`headless=False`) for live viewing.

### Caveats

* Sim taxel forces (Newtons, `tanh`-bounded) are mapped through normalization
  stats computed on **real XELA hardware data** — embeddings are valid but the
  input distribution differs from pretraining; treat absolute embedding values
  cautiously when fine-tuning decisions on them.
* Training used baseline subtraction on real sensor drift; the sim has no
  baseline drift, so that step is a no-op for simulated streams.

## Regenerate the URDF (if the sensor layout changes)


```bash
cd manifeel-isaacgymenvs
python isaacgymenvs/tasks/allegro_kuka/gen_xela_urdf.py
```

## Notes / assumptions

* Uses the **7-DOF** KUKA arm (matching the existing `kuka_allegro` asset), per
  the request that a 7-DOF arm is acceptable.
* Proprioception includes both arm+hand DOFs (46 pos/vel + task state). To match
  Tactile-JEPA's hand-only 16-DOF `joint_angles` exactly, trim the proprio block
  in `compute_full_state` / observation layout.
* The per-taxel distribution is a documented physics-informed approximation of
  the distributed XELA sensing (PhysX exposes per-body, not per-taxel, forces).
  The `build_connectivity_graph()` helper in `xela_sensor_config.py` exposes the
  Tactile-JEPA topology (adjacency + hand-part group ids) for topology-aware
  encoders / masking.
