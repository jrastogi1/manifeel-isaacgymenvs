"""Hybrid data collection for predictor training (Option 3).

Pipeline per environment:
  1. Scripted reach: move fingertips toward fixed object position.
  2. Scripted grasp: close fingers on object.
  3. Random perturbation: once contact detected, apply small random delta actions.
  4. Auto-reset when object drops (z < 0.1) or max episode length hit.

Saves (z_t, a_t, z_{t+1}, done) shards where z = [tactile_latent(192), state(117)].

To run: 
    python3 manifeel-isaacgymenvs/isaacgymenvs/tasks/allegro_kuka/predictor_pipeline/collect_data.py --num_envs 512 --total_steps 10000 --device cuda:0
"""

import os
import sys
import argparse
import numpy as np


# Ensure the manifeel-isaacgymenvs root is on sys.path so that
# tactile_jepa_adapter and predictor_pipeline can be imported.
_MANIFEEL_ROOT = '/home/jrastogi/Documents/cs_projects/manifeel/manifeel-isaacgymenvs'
if _MANIFEEL_ROOT not in sys.path:
    sys.path.insert(0, _MANIFEEL_ROOT)
_TACTILE_REPO = '/home/jrastogi/Documents/cs_projects/manifeel/tactile'
if _TACTILE_REPO not in sys.path:
    sys.path.insert(0, _TACTILE_REPO)

# Isaac Gym requires isaacgym to be imported before torch.
import isaacgym  # noqa: E402  (must come before torch import)

from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_xela import AllegroKukaXela
from omegaconf import OmegaConf

import torch  # noqa: E402  (must come after isaacgym import)

from tactile_jepa_adapter import TactileJEPAEncoder


def load_task_cfg():
    """Load the AllegroKukaXela task config.

    Tries Hydra compose first (handles YAML interpolations), falls back to
    using isaacgymenvs.make() which internally uses the same config.
    """
    try:
        from hydra import compose, initialize
        from hydra.core.hydra_config import HydraConfig

        for _name, _fn in [
            ('eq', lambda x, y: x.lower() == y.lower()),
            ('contains', lambda x, y: x.lower() in y.lower()),
            ('if', lambda pred, a, b: a if pred else b),
            ('resolve_default', lambda default, arg: default if arg == '' else arg),
        ]:
            if not OmegaConf.has_resolver(_name):
                OmegaConf.register_new_resolver(_name, _fn)
        if not OmegaConf.has_resolver('eval'):
            OmegaConf.register_new_resolver('eval', eval)

        if HydraConfig.initialized():
            HydraConfig.instance().clear()

        cfg_dir = "./cfg"
        with initialize(config_path=cfg_dir, version_base=None):
            cfg = compose(config_name="config", overrides=["task=AllegroKukaXela"])
        return cfg.task
    except Exception:
        # Fallback: return None so __init__ uses isaacgymenvs.make()
        return None


def make_env(cfg=None, headless=True):
    """Create env via isaacgymenvs.make, handling config properly."""
    if cfg is None:
        cfg_path = ('/home/jrastogi/Documents/cs_projects/manifeel/'
                     'manifeel-isaacgymenvs/isaacgymenvs/cfg/task/AllegroKukaXela.yaml')
        from hydra import compose, initialize
        from hydra.core.hydra_config import HydraConfig
        OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
        OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
        OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
        OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)
        OmegaConf.register_new_resolver('eval', eval)
        if HydraConfig.initialized():
            HydraConfig.instance().clear()
        with initialize(config_path=cfg_path, version_base=None):
            cfg = compose(config_name="config", overrides=["task=AllegroKukaXela"])
    env_cfg = omegaconf_to_dict(cfg)
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    env = isaacgymenvs.make(**env_cfg, headless=headless, num_envs=1)
    return env


class HybridDataCollector:
    """Option 3 hybrid data collector.

    Per environment:
      Phase 0 (reach):   move arm toward object
      Phase 1 (grasp):   close fingers
      Phase 2 (perturb): random delta actions once contact detected
    """

    def __init__(self, num_envs=2048, device="cuda:0", save_dir="./predictor_data",
                 save_every=50000, contact_threshold=0.5, perturb_scale=0.05,
                 max_episode_steps=300, window_frames=10):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.save_dir = save_dir
        self.save_every = save_every
        self.contact_threshold = contact_threshold
        self.perturb_scale = perturb_scale
        self.max_episode_steps = max_episode_steps
        self.window_frames = window_frames
        os.makedirs(save_dir, exist_ok=True)

        cfg = load_task_cfg()
        if cfg is not None:
            self.env = AllegroKukaXela(
                cfg=cfg,
                rl_device=device,
                sim_device=device,
                graphics_device_id=-1,
                headless=True,
                virtual_screen_capture=False,
                force_render=False,
            )
        else:
            # Fallback: create env via isaacgymenvs.make() which internally
            # loads the same YAML config and resolves all interpolations.
            from isaacgymenvs import make as make_env

            self.env = make_env(
                seed=42,
                task="AllegroKukaXela",
                num_envs=num_envs,
                sim_device=device,
                rl_device=device,
                graphics_device_id=-1,
                headless=True,
                virtual_screen_capture=False,
                force_render=False,
            )

        self.base_full_state_size = self.env.base_full_state_size
        self.taxel_obs_size = self.env.taxel_obs_size
        self.action_dim = self.env.num_allegro_kuka_actions
        self.state_dim = self.base_full_state_size
        self.latent_dim = 192  # JEPA pooled dim

        # Frozen JEPA encoder (real-world pretrained, no fine-tuning)
        self.encoder = TactileJEPAEncoder(device=str(self.device))
        self.encoder.model.eval()
        for p in self.encoder.model.parameters():
            p.requires_grad = False

        self.buffer = []
        self.shard_idx = 0
        self.total_transitions = 0

        # Taxel window: [num_envs, window_frames, 368, 3]
        self.taxel_window = None
        self.phase = None
        self.step_count = None

    def _extract_obs_parts(self, obs):
        """Extract state and taxels from step()/reset() return dict."""
        obs_tensor = obs["obs"] if isinstance(obs, dict) else obs
        state = obs_tensor[:, :self.base_full_state_size]
        taxels = obs_tensor[:, self.base_full_state_size:]
        taxels = taxels.reshape(self.num_envs, 368, 3)
        return state, taxels

    def _init_taxel_window(self, taxels):
        """Initialize window with current taxel frame repeated across all slots."""
        self.taxel_window = taxels.unsqueeze(1).repeat(1, self.window_frames, 1, 1)

    def _update_taxel_window(self, taxels):
        """Slide window: drop oldest frame, append newest."""
        self.taxel_window = torch.cat([
            self.taxel_window[:, 1:, :, :],
            taxels.unsqueeze(1),
        ], dim=1)

    def _encode_taxels(self):
        """Encode current taxel window -> [num_envs, 192] tactile latent."""
        with torch.no_grad():
            out = self.encoder.encode(self.taxel_window)
            z_tac = out["pooled"]  # [B, 192]
        return z_tac

    def _compute_z(self, state, taxels):
        """Compute general latent z = concat(tactile_latent, state)."""
        z_tac = self._encode_taxels()
        z = torch.cat([z_tac, state], dim=-1)  # [num_envs, 192 + 117]
        return z

    def _generate_actions(self, state, taxels):
        """Option 3: scripted reach -> grasp -> random perturbation."""
        actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)

        # Extract object position - works with both AllegroKukaXela env and isaacgymenvs.make() fallback
        if hasattr(self.env, 'object_pos'):
            obj_pos = self.env.object_pos
        elif state.shape[1] >= 72:
            obj_pos = state[:, 69:72]
        else:
            obj_pos = None

        # Extract palm position - works with both env types
        if hasattr(self.env, 'palm_pos'):
            palm_pos = self.env.palm_pos
        elif state.shape[1] >= 49:
            palm_pos = state[:, 46:49]
        else:
            palm_pos = state[:, :3]

        # Ensure phase and step_count are always initialized (prevents state leakage
# between calls or across multiple collect() runs).
        if not hasattr(self, 'phase') or self.phase is None:
            self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            # Always reset phase to 0 at the start of each _generate_actions call
            # to ensure reach_mask includes all environments.
            self.phase.zero_()
        if not hasattr(self, 'step_count') or self.step_count is None:
            self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            self.step_count.zero_()

        phase = self.phase

        # --- Phase 0: reach ------------------------------------------------
        reach_mask = phase == 0
        if reach_mask.any():
            if obj_pos is not None:
                dir_to_obj = obj_pos - palm_pos
                dir_to_obj = dir_to_obj / (
                    torch.norm(dir_to_obj, dim=-1, keepdim=True) + 1e-8
                )
                actions[reach_mask, :3] = dir_to_obj[reach_mask] * 0.5
            actions[reach_mask, 7:] = 0.0
            dist = torch.norm(obj_pos - palm_pos, dim=-1) if obj_pos is not None else torch.zeros_like(dist)
            close_mask = reach_mask & (dist < 0.15)
            phase[close_mask] = 1

        # --- Phase 1: grasp ------------------------------------------------
        grasp_mask = phase == 1
        if grasp_mask.any():
            actions[grasp_mask, :3] = 0.0
            actions[grasp_mask, 7:] = -1.0  # close fingers
            taxel_mag = torch.norm(taxels, dim=-1).sum(dim=-1)
            contact_mask = grasp_mask & (taxel_mag > self.contact_threshold)
            phase[contact_mask] = 2
            self.step_count[contact_mask] = 0

        # --- Phase 2: random perturbation ----------------------------------
        perturb_mask = phase == 2
        if perturb_mask.any():
            rand_actions = torch.randn((perturb_mask.sum(), self.action_dim),
                                       device=self.device) * self.perturb_scale
            actions[perturb_mask] = rand_actions

        return actions

    def collect(self, total_steps=2_000_000):
        """Main data collection loop."""
        obs_dict = self.env.reset()
        obs = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict
        state, taxels = self._extract_obs_parts(obs)
        self._init_taxel_window(taxels)

        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        for global_step in range(total_steps):
            # --- Compute z_t (current latent) ---
            z_t = self._compute_z(state, taxels)           # GPU [num_envs, 309]
            z_t_cpu = z_t.cpu().numpy()                    # move to CPU for storage
            del z_t                                        # free GPU before next compute

            # --- Generate actions (keep as GPU tensor for env.step) ---
            actions = self._generate_actions(state, taxels)  # GPU tensor [num_envs, action_dim]

            # --- Step environment (requires torch.Tensor) ---
            next_obs_dict, reward, done, info = self.env.step(actions)
            next_obs = next_obs_dict["obs"] if isinstance(next_obs_dict, dict) else next_obs_dict
            next_state, next_taxels = self._extract_obs_parts(next_obs)

            actions_cpu = actions.cpu().numpy()            # move to CPU for storage
            del actions

            # --- Compute z_tp1 (next latent) ---
            self._update_taxel_window(next_taxels)
            z_tp1 = self._compute_z(next_state, next_taxels)  # GPU tensor
            z_tp1_cpu = z_tp1.cpu().numpy()                   # CPU for storage
            del z_tp1

            done_cpu = done.cpu().numpy().astype(np.float32)

            # --- Store transition (all CPU numpy) ---
            self.buffer.append({
                "z_t": z_t_cpu,
                "a_t": actions_cpu,
                "z_tp1": z_tp1_cpu,
                "done": done_cpu,
            })
            self.total_transitions += self.num_envs

            # Advance state (keep on GPU)
            state = next_state
            taxels = next_taxels

            # Reset phase/step_count for envs that just reset
            if done.any():
                reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
                self.phase[reset_ids] = 0
                self.step_count[reset_ids] = 0
                self.taxel_window[reset_ids] = next_taxels[reset_ids].unsqueeze(1).repeat(
                    1, self.window_frames, 1, 1
                )

            # Save shard periodically
            if (global_step + 1) % self.save_every == 0:
                self._save_shard()
                self.shard_idx += 1
                self.buffer = []
                print(f"[collect] Step {global_step + 1}/{total_steps}, "
                      f"saved shard {self.shard_idx}, total transitions: {self.total_transitions}")

            if (global_step + 1) % 100 == 0:
                print(f"[collect] Step {global_step + 1}/{total_steps} "
                      f"(transitions: {self.total_transitions})")

        if self.buffer:
            self._save_shard()

        print(f"[collect] Finished. Total transitions: {self.total_transitions}")

    def _save_shard(self):
        if not self.buffer:
            return
        z_t = np.concatenate([b["z_t"] for b in self.buffer], axis=0)
        a_t = np.concatenate([b["a_t"] for b in self.buffer], axis=0)
        z_tp1 = np.concatenate([b["z_tp1"] for b in self.buffer], axis=0)
        done = np.concatenate([b["done"] for b in self.buffer], axis=0)
        shard_path = os.path.join(self.save_dir, f"shard_{self.shard_idx:05d}.npz")
        np.savez(shard_path, z_t=z_t, a_t=a_t, z_tp1=z_tp1, done=done)
        print(f"[collect] Saved shard {self.shard_idx}: {z_t.shape[0]} transitions -> {shard_path}")
        self.buffer = []
        self.shard_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--save_dir", type=str, default="./predictor_data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_every", type=int, default=1024)
    args = parser.parse_args()
    collector = HybridDataCollector(
        num_envs=args.num_envs,
        device=args.device,
        save_dir=args.save_dir,
        save_every=args.save_every,
    )
    collector.collect(total_steps=args.total_steps)
