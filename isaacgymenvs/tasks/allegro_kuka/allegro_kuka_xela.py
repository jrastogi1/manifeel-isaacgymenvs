# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""RL environment: 16-DOF Allegro hand with XELA tactile sensors on a 7-DOF arm.

This task attaches XELA uSkin electronic-skin patches to a 16-DOF Allegro hand
mounted on a 7-DOF KUKA iiwa arm, matching the sensor layout used by the
Tactile-JEPA reference setup (https://huggingface.co/Tactile-JEPA/tactile-jepa-xela):

  * 18 XELA sensor patches (4 fingertips x 30, 11 phalanges x 16, 3 palm x 24)
    -> 368 taxels, each reporting a 3-axis touch reading (X, Y, Z).

Observations concatenate the proprioception state (arm + hand DOF positions and
velocities) with the flattened (368 x 3) taxel tensor, laid out in the same
order as Tactile-JEPA's ``XELA_FLATTEN_ORDER``.

Tactile model (per-taxel 3-axis forces from per-patch PhysX contact forces):
  PhysX reports a net contact force per rigid body, not per taxel. For each of
  the 18 patch bodies we read the net contact force, rotate it into the patch
  local frame (giving local shear-X, shear-Y and normal-Z), and then spatially
  distribute that patch force across the patch's taxels. The effective contact
  point is estimated from the patch wrench; each taxel receives a share of the
  patch force according to a Gaussian falloff in distance from that contact
  point (closer taxels read more). This yields a spatially varying (368, 3)
  tactile image while keeping the rigid-body count low (only +18 bodies/env).
"""

import os
from copy import copy
from os.path import basename

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_reorientation import AllegroKukaReorientation
from isaacgymenvs.tasks.allegro_kuka.xela_sensor_config import (
    NUM_XELA_TAXELS,
    TAXEL_DIM,
    XELA_FLATTEN_ORDER,
    patch_taxel_local_positions,
)
from isaacgymenvs.utils.torch_jit_utils import to_torch, quat_rotate_inverse


class AllegroKukaXela(AllegroKukaReorientation):
    """Allegro hand + XELA tactile sensors on a KUKA arm (reorientation task).

    Inherits the object-reorientation reward / reset machinery from
    ``AllegroKukaReorientation`` (which inherits ``AllegroKukaBase``) and layers
    on the XELA taxel observation. Only the tactile-specific pieces (asset, body
    handles, taxel tensors, observation layout) are overridden/added here.
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # --- XELA-specific config (read before super().__init__) -------------
        env_cfg = cfg["env"]
        self.xela_asset_file = env_cfg["asset"]["kukaAllegro"]

        # Force/tactile scaling knobs (see AllegroKukaXela.yaml).
        self.taxel_force_scale = env_cfg.get("taxelForceScale", 1.0)
        self.taxel_normalize = env_cfg.get("taxelNormalize", True)
        self.taxel_clip = env_cfg.get("taxelClip", 10.0)
        # Gaussian bandwidth (metres) for spatial distribution of patch force.
        self.taxel_sigma = env_cfg.get("taxelSigma", 0.006)
        # Minimum patch force magnitude (N) below which the patch is treated as
        # contact-free (avoids noise-driven taxel activations).
        self.taxel_force_threshold = env_cfg.get("taxelForceThreshold", 1e-3)

        # Size of the flattened (368, 3) taxel observation.
        self.num_xela_taxels = NUM_XELA_TAXELS
        self.taxel_obs_size = NUM_XELA_TAXELS * TAXEL_DIM  # 1104

        # Order of XELA patch link names (matches Tactile-JEPA flatten order).
        self.xela_patch_names = list(XELA_FLATTEN_ORDER.keys())

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        # Frame the viewer on the hand/object workspace. The inherited
        # AllegroKuka camera targets (6, 5, 0), ~7 m from this task's spawn
        # layout (robot base (0, 0.8, 0), object (0, 0, 0.63)), which leaves
        # the robot tiny and off-centre in the frame.
        if self.viewer is not None:
            cam_target = gymapi.Vec3(0.0, 0.3, 0.55)
            cam_pos = gymapi.Vec3(1.3, -1.3, 1.3)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Extend the observation/state buffers with the flattened taxel tensor
        # and precompute the per-taxel local geometry used to distribute patch
        # contact forces. Runs after the base env (sim, bodies) is built.
        self._setup_xela_observation()

    # ------------------------------------------------------------------
    # XELA tactile setup
    # ------------------------------------------------------------------
    def _setup_xela_observation(self):
        """Allocate the extended obs buffer and precompute taxel geometry.

        The base class sizes ``self.obs_buf`` / ``self.states_buf`` to the
        proprioception-only ``full_state_size``. Here we grow both buffers by
        ``taxel_obs_size`` and append the (368, 3) taxel block after the base
        state, so the RL observation is ``[proprio, taxels]``.
        """
        # 1) grow observation / state buffers ------------------------------
        # NOTE: we intentionally keep ``self.full_state_size`` at its base value
        # so the inherited ``compute_full_state`` (and its size assert) keeps
        # working. We only enlarge the flat obs/state buffers and append the
        # taxel block after the base state.
        self.base_full_state_size = self.full_state_size

        self.num_observations += self.taxel_obs_size
        self.num_states += self.taxel_obs_size

        self.cfg["env"]["numObservations"] = self.num_observations
        self.cfg["env"]["numStates"] = self.num_states

        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations), dtype=torch.float, device=self.device
        )
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), dtype=torch.float, device=self.device
        )
        # gym spaces are read by rl_games to size the network input.
        import gym

        self.obs_space = gym.spaces.Box(
            np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf
        )
        self.state_space = gym.spaces.Box(
            np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf
        )

        # 2) per-taxel local geometry (patch-local frame) -------------------
        # For every taxel we store: which patch it belongs to (patch id), and
        # its (x, y) position on that patch's sensor grid. These are fixed and
        # let us spatially distribute each patch's contact force.
        patch_ids = []
        local_xy = []
        patch_centroids = []
        for patch_idx, name in enumerate(self.xela_patch_names):
            positions = patch_taxel_local_positions(name)  # (n_taxels, 3)
            n = positions.shape[0]
            patch_ids.append(np.full((n,), patch_idx, dtype=np.int64))
            local_xy.append(positions[:, :2].astype(np.float32))
            patch_centroids.append(positions[:, :2].mean(axis=0))

        self.taxel_patch_id = to_torch(np.concatenate(patch_ids), dtype=torch.long, device=self.device)
        self.taxel_local_xy = to_torch(np.concatenate(local_xy, axis=0), device=self.device)  # (368, 2)
        self.patch_centroids = to_torch(np.stack(patch_centroids, axis=0), device=self.device)  # (18, 2)

        # Gaussian inverse-variance for the spatial falloff (shared sigma).
        self.inv_two_sigma_sq = 1.0 / (2.0 * self.taxel_sigma ** 2)

        # 3) scratch buffer for the (num_envs, 368, 3) taxel tensor ---------
        self.taxels = torch.zeros(
            (self.num_envs, self.num_xela_taxels, TAXEL_DIM), dtype=torch.float, device=self.device
        )

        # 4) acquire the per-body net contact force tensor -------------------
        self.acquire_contact_force_tensor()

        print(
            f"[AllegroKukaXela] obs = proprio({self.base_full_state_size}) "
            f"+ taxels({self.taxel_obs_size}) = {self.num_observations}; "
            f"taxels={self.num_xela_taxels}x{TAXEL_DIM}"
        )

    # ------------------------------------------------------------------
    # Environment construction (XELA override)
    # ------------------------------------------------------------------
    def _create_envs(self, num_envs, spacing, num_per_row):
        """Same as AllegroKukaBase._create_envs with two XELA-specific changes:

        1. ``collapse_fixed_joints = False`` so the 18 uSkin patch links remain
           distinct rigid bodies whose per-body contact force we can read.
        2. Resolve the per-asset rigid-body index of every XELA patch link.
        """
        import tempfile

        from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import populate_dof_properties
        from isaacgymenvs.utils.torch_jit_utils import get_axis_params

        if self.should_load_initial_states:
            self.load_initial_states()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")

        object_asset_root = asset_root
        tmp_assets_dir = tempfile.TemporaryDirectory()
        self.object_asset_files, self.object_asset_scales = self._main_object_assets_and_scales(
            object_asset_root, tmp_assets_dir.name
        )

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        # XELA: keep the sensor-patch links as distinct rigid bodies.
        asset_options.collapse_fixed_joints = False
        # Gravity is disabled on the hand/arm: the position controller holds the
        # hand at commanded poses and presses fingers into objects, generating
        # contact forces for the XELA taxels. Enabling gravity here would collapse
        # the arm (finger effort is only 0.35 N·m).
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading asset {self.hand_arm_asset_file} from {asset_root}")
        allegro_kuka_asset = self.gym.load_asset(self.sim, asset_root, self.hand_arm_asset_file, asset_options)
        print(f"Loaded asset {allegro_kuka_asset}")

        self.num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(allegro_kuka_asset)
        self.num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(allegro_kuka_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(allegro_kuka_asset)
        assert (
            self.num_hand_arm_dofs == num_hand_arm_dofs
        ), f"Number of DOFs in asset {allegro_kuka_asset} is {num_hand_arm_dofs}, but {self.num_hand_arm_dofs} was expected"

        max_agg_bodies = self.num_hand_arm_bodies
        max_agg_shapes = self.num_hand_arm_shapes

        allegro_rigid_body_names = [
            self.gym.get_asset_rigid_body_name(allegro_kuka_asset, i) for i in range(self.num_hand_arm_bodies)
        ]
        print(f"Allegro num rigid bodies: {self.num_hand_arm_bodies}")
        print(f"Allegro rigid bodies: {allegro_rigid_body_names}")

        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_kuka_asset)

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.allegro_sensors = []
        allegro_sensor_pose = gymapi.Transform()

        for i in range(self.num_hand_arm_dofs):
            self.arm_hand_dof_lower_limits.append(allegro_hand_dof_props["lower"][i])
            self.arm_hand_dof_upper_limits.append(allegro_hand_dof_props["upper"][i])

        self.arm_hand_dof_lower_limits = to_torch(self.arm_hand_dof_lower_limits, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(self.arm_hand_dof_upper_limits, device=self.device)

        allegro_pose = gymapi.Transform()
        allegro_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(0.0, 0.8, 0)
        allegro_pose.r = gymapi.Quat(0, 0, 0, 1)

        object_assets, object_rb_count, object_shapes_count = self._load_main_object_asset()
        max_agg_bodies += object_rb_count
        max_agg_shapes += object_shapes_count

        # load auxiliary objects
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = False
        table_asset_options.fix_base_link = True
        table_asset = self.gym.load_asset(self.sim, asset_root, self.asset_files_dict["table"], table_asset_options)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = allegro_pose.p.x
        table_pose_dy, table_pose_dz = -0.8, 0.38
        table_pose.p.y = allegro_pose.p.y + table_pose_dy
        table_pose.p.z = allegro_pose.p.z + table_pose_dz

        table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)
        max_agg_bodies += table_rb_count
        max_agg_shapes += table_shapes_count

        additional_rb, additional_shapes = self._load_additional_assets(object_asset_root, allegro_pose)
        max_agg_bodies += additional_rb
        max_agg_shapes += additional_shapes

        # set up object and goal positions
        self.object_start_pose = self._object_start_pose(allegro_pose, table_pose_dy, table_pose_dz)

        self.allegro_hands = []
        self.envs = []

        object_init_state = []

        self.allegro_hand_indices = []
        object_indices = []
        object_scales = []
        object_keypoint_offsets = []

        self.allegro_fingertip_handles = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in self.allegro_fingertips
        ]

        self.allegro_palm_handle = self.gym.find_asset_rigid_body_index(allegro_kuka_asset, "iiwa7_link_7")

        # XELA: resolve the rigid-body index of each of the 18 sensor patches.
        self.xela_patch_body_ids = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in self.xela_patch_names
        ]
        for name, bid in zip(self.xela_patch_names, self.xela_patch_body_ids):
            assert bid >= 0, f"XELA patch link {name!r} not found in asset {self.hand_arm_asset_file}"
        print(f"XELA patch body ids: {self.xela_patch_body_ids}")

        # this rely on the fact that objects are added right after the arms in terms of create_actor()
        self.object_rb_handles = list(range(self.num_hand_arm_bodies, self.num_hand_arm_bodies + object_rb_count))

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            allegro_actor = self.gym.create_actor(env_ptr, allegro_kuka_asset, allegro_pose, "allegro", i, -1, 0)

            populate_dof_properties(allegro_hand_dof_props, self.dof_params, self.num_arm_dofs, self.num_hand_dofs)

            self.gym.set_actor_dof_properties(env_ptr, allegro_actor, allegro_hand_dof_props)
            allegro_hand_idx = self.gym.get_actor_index(env_ptr, allegro_actor, gymapi.DOMAIN_SIM)
            self.allegro_hand_indices.append(allegro_hand_idx)

            if self.obs_type == "full_state":
                if self.with_fingertip_force_sensors:
                    for ft_handle in self.allegro_fingertip_handles:
                        env_sensors = [self.gym.create_force_sensor(env_ptr, ft_handle, allegro_sensor_pose)]
                        self.allegro_sensors.append(env_sensors)

                if self.with_dof_force_sensors:
                    self.gym.enable_actor_dof_force_sensors(env_ptr, allegro_actor)

            # add object
            object_asset_idx = i % len(object_assets)
            object_asset = object_assets[object_asset_idx]

            object_handle = self.gym.create_actor(env_ptr, object_asset, self.object_start_pose, "object", i, 0, 0)
            object_init_state.append(
                [
                    self.object_start_pose.p.x,
                    self.object_start_pose.p.y,
                    self.object_start_pose.p.z,
                    self.object_start_pose.r.x,
                    self.object_start_pose.r.y,
                    self.object_start_pose.r.z,
                    self.object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            object_indices.append(object_idx)

            object_scale = self.object_asset_scales[object_asset_idx]
            object_scales.append(object_scale)
            object_offsets = []
            for keypoint in self.keypoints_offsets:
                keypoint = copy(keypoint)
                for coord_idx in range(3):
                    keypoint[coord_idx] *= object_scale[coord_idx] * self.object_base_size * self.keypoint_scale / 2
                object_offsets.append(keypoint)

            object_keypoint_offsets.append(object_offsets)

            # table object
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table_object", i, 0, 0)
            table_object_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)

            # task-specific objects (i.e. goal object for reorientation task)
            self._create_additional_objects(env_ptr, env_idx=i, object_asset_idx=object_asset_idx)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_actor)

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        object_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[0], object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.allegro_fingertip_handles = to_torch(self.allegro_fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.allegro_hand_indices = to_torch(self.allegro_hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(object_indices, dtype=torch.long, device=self.device)

        self.object_scales = to_torch(object_scales, dtype=torch.float, device=self.device)
        self.object_keypoint_offsets = to_torch(object_keypoint_offsets, dtype=torch.float, device=self.device)

        self._after_envs_created()

        try:
            # by this point we don't need the temporary folder for procedurally generated assets
            tmp_assets_dir.cleanup()
        except Exception:
            pass

    def _load_main_object_asset(self):
        """Load manipulated object with gravity enabled so it settles onto the table."""
        object_asset_options = gymapi.AssetOptions()
        # Explicitly enable gravity so objects fall onto the table instead of
        # floating in mid-air (default is False, but be explicit).
        object_asset_options.disable_gravity = False
        object_assets = []
        for object_asset_file in self.object_asset_files:
            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = basename(object_asset_file)
            object_asset_ = self.gym.load_asset(self.sim, object_asset_dir, object_asset_fname, object_asset_options)
            object_assets.append(object_asset_)
        object_rb_count = self.gym.get_asset_rigid_body_count(
            object_assets[0]
        )
        object_shapes_count = self.gym.get_asset_rigid_shape_count(
            object_assets[0]
        )
        return object_assets, object_rb_count, object_shapes_count

    # ------------------------------------------------------------------
    # Tactile acquisition
    # ------------------------------------------------------------------
    def acquire_contact_force_tensor(self):
        """Acquire the per-rigid-body net contact force tensor.

        Shape after wrapping: (num_envs, num_bodies, 3). We read the rows that
        correspond to the 18 XELA patch bodies to obtain each patch's 3-axis
        contact force in the world frame.
        """
        if not hasattr(self, "contact_force_net"):
            _contact_force_net = self.gym.acquire_net_contact_force_tensor(self.sim)
            self.contact_force_net = gymtorch.wrap_tensor(_contact_force_net).view(self.num_envs, -1, 3)

    def refresh_contact_forces(self):
        """Refresh and slice out the XELA patch forces + their world quats."""
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # (num_envs, 18, 3) net contact force of each patch body, world frame.
        patch_ids = to_torch(self.xela_patch_body_ids, dtype=torch.long, device=self.device)
        self.xela_patch_forces = self.contact_force_net[:, patch_ids, :]

        # world-frame orientation of each patch body, used to rotate the force
        # into the patch-local (sensor) frame.
        self.xela_patch_quats = self.rigid_body_states[:, patch_ids, 3:7]

    def compute_taxels(self):
        """Build the (num_envs, 368, 3) taxel tensor from patch contact forces.

        For each patch:
          * rotate its net contact force into the patch-local frame, giving the
            local 3-axis reading (shear-X, shear-Y, normal-Z);
          * estimate an effective contact point in the patch plane from the
            local force (shear bias), defaulting to the patch centroid;
          * distribute the local force across the patch's taxels with a Gaussian
            falloff in distance from that contact point (weights sum to 1), so
            the taxel forces sum back to the patch force and nearby taxels read
            more than distant ones.
        """
        # local 3-axis force per patch: (num_envs, 18, 3)
        # quat_rotate_inverse expects (N, 4) and (N, 3); flatten the patch dim
        # into the batch dim, then restore the (num_envs, 18, 3) shape.
        num_patches = len(self.xela_patch_names)
        quats_flat = self.xela_patch_quats.reshape(self.num_envs * num_patches, 4)
        forces_flat = self.xela_patch_forces.reshape(self.num_envs * num_patches, 3)
        patch_force_local = quat_rotate_inverse(quats_flat, forces_flat).view(
            self.num_envs, num_patches, 3
        )

        self.taxels.zero_()

        # process one patch at a time (18 iterations) to keep indexing simple.
        for patch_idx in range(len(self.xela_patch_names)):
            taxel_idx = (self.taxel_patch_id == patch_idx).nonzero(as_tuple=False).squeeze(-1)  # (n_taxels,)
            n_taxels = taxel_idx.shape[0]
            if n_taxels == 0:
                continue

            local_xy = self.taxel_local_xy[taxel_idx]  # (n_taxels, 2) patch-local
            force = patch_force_local[:, patch_idx, :]  # (num_envs, 3)

            # magnitude gate: skip patches with (near-)zero contact.
            force_mag = torch.norm(force, dim=-1)  # (num_envs,)
            active = (force_mag > self.taxel_force_threshold).to(self.taxels.dtype)  # (num_envs,)

            # Effective contact point in the patch plane. PhysX net-contact gives
            # the force only (not the application point), so we infer a contact
            # location from the local shear direction: shear (x, y) normalized by
            # the normal magnitude gives a directional bias; centred contact
            # (no shear) falls back to the patch centroid.
            fx = force[:, 0:1]
            fy = force[:, 1:2]
            fz = force[:, 2:3].abs() + 1e-6
            bias_scale = 0.01  # metres per (unit shear / unit normal)
            contact_xy = torch.cat([fx, fy], dim=-1) / fz * bias_scale  # (num_envs, 2)
            centroid = self.patch_centroids[patch_idx].unsqueeze(0)  # (1, 2)
            contact_xy = contact_xy + centroid
            max_off = 0.02
            contact_xy = torch.clamp(contact_xy, centroid - max_off, centroid + max_off)

            # Gaussian weights over taxels: (num_envs, n_taxels)
            # squared distance between each env's contact point and each taxel.
            diff = contact_xy.unsqueeze(1) - local_xy.unsqueeze(0)  # (num_envs, n_taxels, 2)
            d2 = (diff * diff).sum(dim=-1)  # (num_envs, n_taxels)
            weights = torch.exp(-self.inv_two_sigma_sq * d2)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            # distribute the local 3-axis force across taxels
            taxel_force = weights.unsqueeze(-1) * force.unsqueeze(1)  # (num_envs, n_taxels, 3)
            taxel_force = taxel_force * active.view(self.num_envs, 1, 1)

            self.taxels[:, taxel_idx, :] = taxel_force

        # scale + squash into a stable range for learning
        self.taxels.mul_(self.taxel_force_scale)
        if self.taxel_normalize:
            self.taxels = torch.tanh(self.taxels)
        self.taxels.clamp_(-self.taxel_clip, self.taxel_clip)
        return self.taxels

    # ------------------------------------------------------------------
    # Observations: [proprioception, taxels]
    # ------------------------------------------------------------------
    def compute_observations(self):
        """Base proprioception + task state, then append the flattened taxels.

        The inherited ``compute_observations`` refreshes the DOF / root / rigid
        body state tensors and fills ``self.obs_buf[:, :base_full_state_size]``
        via ``compute_full_state``. We then refresh the contact-force tensor,
        build the (368, 3) taxel tensor, and write it (flattened) right after
        the base state. The observation layout is therefore::

            [ dof_pos(23), dof_vel(23), ..., task_state..., taxels(368*3) ]

        matching Tactile-JEPA's proprioception + taxel observation structure.
        """
        obs_buf, reward_obs_ofs = super().compute_observations()

        # refresh contact forces (needs the rigid-body states already refreshed
        # by the base compute_observations) and build the taxel tensor.
        self.refresh_contact_forces()
        taxels = self.compute_taxels()  # (num_envs, 368, 3)

        ofs = self.base_full_state_size
        self.obs_buf[:, ofs : ofs + self.taxel_obs_size] = taxels.reshape(self.num_envs, self.taxel_obs_size)
        self.states_buf[:, ofs : ofs + self.taxel_obs_size] = taxels.reshape(self.num_envs, self.taxel_obs_size)

        return self.obs_buf, reward_obs_ofs






