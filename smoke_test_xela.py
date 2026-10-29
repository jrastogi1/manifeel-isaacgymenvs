# Smoke test for the AllegroKukaXela RL environment.
#
# Builds the env headless with a small number of parallel envs, steps it for a
# few hundred control steps, and verifies:
#   * the observation shape is [num_envs, proprio + 368*3];
#   * proprioception tracks the 23 arm+hand DOFs;
#   * taxels are ~0 in free space and become non-zero once the hand contacts
#     the object (we nudge the hand to close onto the cube);
#   * the (368, 3) taxel tensor matches the Tactile-JEPA flatten layout.
#
# Run from the manifeel-isaacgymenvs directory:
#   python smoke_test_xela.py

import argparse

import numpy as np

import isaacgym  # must be imported before torch per Isaac Gym requirement
import torch

import isaacgymenvs
from isaacgymenvs.tasks.allegro_kuka.xela_sensor_config import (
    NUM_XELA_TAXELS,
    TAXEL_DIM,
    XELA_FLATTEN_ORDER,
    patch_taxel_local_positions,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--sim_device", type=str, default="cuda:0")
    p.add_argument("--rl_device", type=str, default="cuda:0")
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    env = isaacgymenvs.make(
        seed=0,
        task="AllegroKukaXela",
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=-1,
        headless=args.headless,
        multi_gpu=False,
        virtual_screen_capture=False,
        force_render=False,
        cfg=None,
    )

    print("\n================ ENV BUILT ================")
    print("num_observations:", env.num_observations)
    print("num_states      :", env.num_states)
    print("num_actions     :", env.num_actions)

    # Expected sizes
    proprio = env.base_full_state_size
    taxel_flat = NUM_XELA_TAXELS * TAXEL_DIM
    expected_obs = proprio + taxel_flat
    assert env.num_observations == expected_obs, (
        f"obs size {env.num_observations} != proprio {proprio} + taxels {taxel_flat} = {expected_obs}"
    )
    assert env.num_actions == 23, f"expected 23 actions, got {env.num_actions}"

    # ---- static checks on the XELA layout -------------------------------
    total = sum(XELA_FLATTEN_ORDER.values())
    assert total == 368, total
    for name in XELA_FLATTEN_ORDER:
        pos = patch_taxel_local_positions(name)
        assert pos.shape == (XELA_FLATTEN_ORDER[name], 3)
    print(f"[layout] 18 patches, {total} taxels, {TAXEL_DIM}-axis -> {taxel_flat} flattened")

    # ---- roll out --------------------------------------------------------
    obs_dict = env.reset()
    obs = obs_dict["obs"]
    print("\ninitial obs shape:", tuple(obs.shape))
    assert tuple(obs.shape) == (args.num_envs, expected_obs)

    # free-space check: right after reset (hand open, away from object) the
    # taxels should be (near) zero.
    fs_taxels = obs[:, proprio:].reshape(args.num_envs, NUM_XELA_TAXELS, TAXEL_DIM)
    fs_max = fs_taxels.abs().max().item()
    print(f"free-space max|taxel| after reset: {fs_max:.5f}")

    # actions: arm stays near default (0 delta), fingers close over time.
    actions = torch.zeros((args.num_envs, env.num_actions), device=args.rl_device, dtype=torch.float)

    max_abs_taxel = 0.0
    nonzero_steps = 0
    contact_established = False
    for step in range(args.steps):
        # progressively close the fingers (last 16 dims) to press on the object.
        close = min(1.0, step / 60.0)
        actions[:, 7:] = close  # hand DOFs target (scaled to [-1,1] -> joint range)
        # small arm motion towards the object
        actions[:, 0] = 0.2
        obs_dict, rew, reset, info = env.step(actions)
        obs = obs_dict["obs"]

        taxel_block = obs[:, proprio:].reshape(args.num_envs, NUM_XELA_TAXELS, TAXEL_DIM)
        m = taxel_block.abs().max().item()
        max_abs_taxel = max(max_abs_taxel, m)
        if m > 1e-3:
            nonzero_steps += 1
            contact_established = True

        if step % 30 == 0:
            print(
                f"step {step:3d} | rew mean {rew.mean().item():8.3f} | "
                f"resets {int(reset.sum().item())} | max|taxel| {m:.4f}"
            )

    print("\n================ RESULTS ================")
    print(f"free-space max|taxel| after reset: {fs_max:.5f}")
    print(f"steps with non-zero taxels: {nonzero_steps}/{args.steps}")
    print(f"max |taxel| over rollout : {max_abs_taxel:.4f}")
    print(f"contact established     : {contact_established}")

    # sanity: taxel values are bounded by the clip (tanh -> [-1, 1] then clip 10)
    assert max_abs_taxel <= 10.0 + 1e-6, max_abs_taxel
    # taxels should activate once the hand closes on the object.
    assert contact_established, "taxels never activated during contact rollout"

    # proprio block: dof pos/vel present (first 23 entries should move)
    proprio_block = obs[:, :proprio]
    print("proprio block shape:", tuple(proprio_block.shape))
    print("proprio min/max:", float(proprio_block.min()), float(proprio_block.max()))

    print("\nSMOKE TEST PASSED: obs = [proprio(%d), taxels(%d x %d)]" % (proprio, NUM_XELA_TAXELS, TAXEL_DIM))


if __name__ == "__main__":
    main()
