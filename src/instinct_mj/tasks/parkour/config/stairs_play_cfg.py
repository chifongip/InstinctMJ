"""Stairs-only keyboard playback shared by G1 and Agibot X2."""

from __future__ import annotations

import copy
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import mujoco
import torch

from instinct_mj.tasks.parkour.mdp.commands.commands_cfg import KeyboardVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

_TERRAIN_SIZE = (8.0, 8.0)
_ENTRANCE_BORDER_WIDTH = 1.0


def _entrance_robot_spec(
    spec_fn: Callable[[], mujoco.MjSpec],
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
) -> mujoco.MjSpec:
    """Put the XML default pose at the entrance before simulation allocation."""
    spec = spec_fn()
    root_bodies = [
        body for body in spec.bodies if any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in body.joints)
    ]
    if len(root_bodies) != 1:
        raise ValueError("Stairs playback requires a robot with exactly one floating base.")
    root_bodies[0].pos = pos
    root_bodies[0].quat = rot
    return spec


def reset_stairs_root_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Spawn on the level entrance facing the ascending stairs (+X)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    state = robot.data.default_root_state[env_ids].clone()
    # Terrain origins lie on the raised central landing. The entrance border
    # is at world Z=0, so only apply the origin's horizontal translation.
    state[:, :2] += env.scene.env_origins[env_ids, :2]
    state[:, 7:13] = 0.0
    robot.write_root_link_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(state[:, 7:13], env_ids=env_ids)


def configure_stairs_play(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Retain policy inputs while replacing terrain and automatic commands."""
    terrain = cfg.scene.terrain
    generator = terrain.terrain_generator
    stairs = copy.deepcopy(generator.sub_terrains["pyramid_stairs"])
    stairs.proportion = 1.0
    stairs.step_height_range = (0.12, 0.12)
    stairs.step_width = 0.3
    stairs.platform_width = 2.5
    stairs.border_width = _ENTRANCE_BORDER_WIDTH
    stairs.perlin_cfg = None
    stairs.wall_prob = [0.0, 0.0, 0.0, 0.0]
    stairs.flat_patch_sampling = {}
    generator.sub_terrains = {"pyramid_stairs": stairs}
    generator.size = _TERRAIN_SIZE
    generator.num_rows = 1
    generator.num_cols = 1
    terrain.max_init_terrain_level = 0
    cfg.scene.num_envs = 1
    robot = cfg.scene.entities["robot"]
    # Also place the model's initial state at the entrance: simulation buffers
    # are allocated before the first reset, and a robot inside the landing
    # would generate excessive contacts during that initial forward pass.
    entrance_x = -0.5 * _TERRAIN_SIZE[0] + 0.5 * _ENTRANCE_BORDER_WIDTH
    robot.init_state.pos = (entrance_x, 0.0, robot.init_state.pos[2])
    robot.spec_fn = partial(_entrance_robot_spec, robot.spec_fn, robot.init_state.pos, robot.init_state.rot)
    cfg.curriculum = {}
    cfg.episode_length_s = 3600.0
    cfg.commands = {"base_velocity": KeyboardVelocityCommandCfg()}
    cfg.terminations.pop("dataset_exhausted", None)
    cfg.events["reset_base"].func = reset_stairs_root_state
    cfg.events["reset_base"].params = {}
    for sensor in cfg.scene.sensors:
        if sensor.name == "leg_volume_points":
            sensor.debug_vis = False
    cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
    cfg.viewer.entity_name = "robot"
    cfg.viewer.body_name = None
    cfg.viewer.distance = 5.0
    cfg.viewer.azimuth = 135.0
    cfg.viewer.elevation = -25.0
    return cfg
