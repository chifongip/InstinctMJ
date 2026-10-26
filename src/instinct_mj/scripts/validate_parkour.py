"""Bounded parkour data, curriculum, training and checkpoint preflight.

Run on the training server without --num-envs to validate the production batch.
A smaller --num-envs only changes this validation process, never task defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
import yaml
from instinct_rl.runners import OnPolicyRunner
from mjlab.utils.os import dump_yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import instinct_mj.tasks  # noqa: F401
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_vecenv_cls
from instinct_mj.utils.dict import class_to_dict


def _finite(value, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite values in {name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite(item, f"{name}[{index}]")


def audit_motions(cfg) -> dict:
    """Check selected files, conventions and hard limits without changing data."""
    reference = next(sensor for sensor in cfg.scene.sensors if sensor.name == "motion_reference")
    model = mujoco.MjModel.from_xml_path(reference.robot_model_path)
    names = [model.joint(index).name for index in range(1, model.njnt)]
    limits = cfg.rewards["rewards"]["dof_vel_limits"].params["velocity_limits"]
    velocity_limits = np.array([
        next(value for pattern, value in limits.items() if re.fullmatch(pattern, name)) for name in names
    ])
    permutation = np.array(reference.symmetric_augmentation_joint_mapping)
    signs = np.array(reference.symmetric_augmentation_joint_reverse_buf)
    counts = {"clips": 0, "frames": 0, "transitions": 0, "overspeed_transitions": 0}
    max_speed_ratio = 0.0
    for buffer in reference.motion_buffers.values():
        root = Path(buffer.path)
        selection = yaml.safe_load(Path(buffer.filtered_motion_selection_filepath).read_text())
        files = selection["selected_files"]
        if not files or len(files) != len(set(files)):
            raise ValueError("Motion selection must contain unique files")
        weights = np.array(selection.get("motion_weights", np.ones(len(files))))
        if weights.shape != (len(files),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Motion weights must be finite, positive and aligned with selected files")
        for relative in files:
            with np.load(root / relative, allow_pickle=False) as data:
                q = data["joint_pos"]
                xyz, quat = data["base_pos_w"], data["base_quat_w"]
                fps = float(data["framerate"])
                if data["joint_names"].tolist() != names:
                    raise ValueError(f"Joint ordering differs from the model: {relative}")
                if len(q) < reference.num_frames or q.shape != (len(q), len(names)):
                    raise ValueError(f"Motion is too short or has the wrong joint dimensions: {relative}")
                if xyz.shape != (len(q), 3) or quat.shape != (len(q), 4):
                    raise ValueError(f"Wrong root pose dimensions: {relative}")
                if not math.isfinite(fps) or fps <= 0 or not all(np.isfinite(v).all() for v in (q, xyz, quat)):
                    raise ValueError(f"Invalid motion values: {relative}")
                if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-5):
                    raise ValueError(f"Non-unit root quaternion: {relative}")
                for positions in (q, q[:, permutation] * signs):
                    if np.any(positions < model.jnt_range[1:, 0] - 1e-6) or np.any(
                        positions > model.jnt_range[1:, 1] + 1e-6
                    ):
                        raise ValueError(f"Original or mirrored motion violates joint limits: {relative}")
                ratios = np.abs(np.diff(q, axis=0)) * fps / velocity_limits
                max_speed_ratio = max(max_speed_ratio, float(ratios.max()))
                counts["clips"] += 1
                counts["frames"] += len(q)
                counts["transitions"] += len(q) - 1
                counts["overspeed_transitions"] += int(np.any(ratios > 1, axis=1).sum())
    return {
        **counts,
        "max_joint_speed_ratio": max_speed_ratio,
        "overspeed_fraction": counts["overspeed_transitions"] / counts["transitions"],
        "root_body": model.body(1).name,
        "note": "Speed outliers are diagnostics for AMP style data, not direct joint tracking commands.",
    }


def _validate_curriculum(env, vec, cfg, args, report: dict) -> None:
    """Check terrain metadata, command assignment, curriculum and every tile."""
    terrain_cfg = cfg.scene.terrain.terrain_generator
    terrain = env.scene.terrain
    ids = torch.arange(env.num_envs, device=args.device)
    columns = terrain.terrain_types.clone()
    rows, cols = terrain.terrain_origins.shape[:2]
    report["terrain_shape"] = [rows, cols]
    if set(columns.tolist()) != set(range(cols)):
        raise ValueError("Not all generated terrain columns were assigned")
    command = env.command_manager.get_term("base_velocity")
    counts = torch.bincount(columns, minlength=cols)
    report["terrain_spawn_counts"] = counts.tolist()
    for col, (name, sub_cfg) in enumerate(terrain_cfg.sub_terrains.items()):
        selected = columns == col
        for component in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            value = getattr(command, component + "_range")[selected]
            expected = torch.tensor(command.cfg.velocity_ranges[name][component], device=args.device)
            torch.testing.assert_close(value, expected.expand_as(value))
        if not torch.all(command.random_velocity_indices[selected] == (name in command.cfg.random_velocity_terrain)):
            raise ValueError(f"Wrong random-velocity assignment for {name}")
        for row in range(rows):
            tile_cfg = terrain.terrain_generator.get_subterrain_cfg(row, col)
            if type(tile_cfg) is not type(sub_cfg):
                raise ValueError(f"Wrong terrain metadata at ({row}, {col})")
            low, high = terrain_cfg.difficulty_range
            expected_difficulty = low + (high - low) * row / max(rows - 1, 1)
            if not math.isclose(tile_cfg.difficulty, expected_difficulty):
                raise ValueError("Wrong difficulty in per-tile terrain metadata")
    for geom, tile_cfg in terrain._iter_hfield_geoms_with_subterrain_cfgs():
        row = int(math.floor(geom.pos[0] / terrain_cfg.size[0] + rows / 2))
        col = int(math.floor(geom.pos[1] / terrain_cfg.size[1] + cols / 2))
        if tile_cfg is not terrain.terrain_generator.get_subterrain_cfg(row, col):
            raise ValueError("Heightfield metadata does not match its world-space tile")
    # Check both curriculum directions independently of an untrained policy.
    terrain.terrain_levels[:2] = 1
    command.metrics["tracking_exp_vel_xy"][:2] = torch.tensor([0.7, 0.1], device=args.device)
    command.metrics["tracking_exp_vel_yaw"][:2] = 1.0
    term = cfg.curriculum["terrain_levels"]
    term.func(env, ids[:2], **term.params)
    if terrain.terrain_levels[:2].tolist() != [2, 0]:
        raise ValueError("Terrain curriculum did not advance and retreat as expected")
    report["curriculum_up_down"] = True
    covered = set()
    ray_data = mujoco.MjData(env.sim.mj_model)
    mujoco.mj_forward(env.sim.mj_model, ray_data)
    terrain_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    hit_id = np.array([-1], dtype=np.int32)
    # Exercise both sides of obstacles and the central spawn platforms.
    for level in range(rows):
        for offset in (-2.8, 0.0, 2.8):
            terrain.terrain_types[:] = columns
            terrain.terrain_levels[:] = level
            terrain.env_origins[:] = terrain.terrain_origins[level, columns]
            # Keep the reset curriculum neutral while choosing a test tile.
            command.metrics["tracking_exp_vel_xy"].fill_(0.45)
            vec.reset()
            if not torch.all(terrain.terrain_levels == level):
                raise ValueError("Reset unexpectedly changed the requested terrain difficulty")
            root = env.scene["robot"].data.root_link_pose_w.clone()
            root[:, 0] += offset
            # Raised/inverted terrains are not flat away from the spawn
            # platform. Place the robot above the local surface rather
            # than teleporting it inside an obstacle at the platform Z.
            for index, xy in enumerate(root[:, :2].cpu().numpy()):
                heights = []
                for dx, dy in ((0, 0), (-0.15, -0.15), (-0.15, 0.15), (0.15, -0.15), (0.15, 0.15)):
                    distance = mujoco.mj_ray(
                        env.sim.mj_model,
                        ray_data,
                        np.array([xy[0] + dx, xy[1] + dy, 100.0]),
                        np.array([0.0, 0.0, -1.0]),
                        terrain_groups,
                        1,
                        -1,
                        hit_id,
                    )
                    if distance >= 0:
                        heights.append(100.0 - distance)
                if not heights:
                    raise ValueError("No terrain surface below the test spawn")
                root[index, 2] += max(heights) - terrain.env_origins[index, 2]
            env.scene["robot"].write_root_link_pose_to_sim(root)
            env.sim.forward()
            covered.update((level, column) for column in columns.tolist())
            for step in range(args.steps_per_level):
                action = torch.zeros(env.num_envs, vec.num_actions, device=args.device)
                if step % 3 == 1:
                    action.uniform_(-0.05, 0.05)
                vec.step(action)
        print(f"Validated terrain level {level + 1}/{rows}", flush=True)
    report["covered_terrain_tiles"] = len(covered)
    if len(covered) != rows * cols:
        raise ValueError("Incomplete terrain coverage")


def _validate_training(vec, agent, args, report: dict) -> None:
    """Exercise learning, checkpoint restoration, resume and inference."""
    runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(args.output), device=args.device)
    try:
        runner.add_git_repo_to_log(__file__)
        runner.learn(args.iterations)
        checkpoint = args.output / f"model_{runner.current_learning_iteration}.pt"
        state = torch.load(checkpoint, map_location=args.device, weights_only=True)
        _finite(state, "checkpoint")
        # Make a no-op loader fail this check: both model and optimizer values
        # must actually be restored from disk before the resumed update.
        with torch.no_grad():
            for module in (runner.alg.actor_critic, runner.alg.discriminator):
                next(module.parameters()).add_(1.0)
            for optimizer in (runner.alg.optimizer, runner.alg.discriminator_optimizer):
                next(iter(optimizer.state.values()))["exp_avg"].add_(1.0)
        runner.load(str(checkpoint))
        for key, value in runner.alg.state_dict().items():
            # Adam keeps its scalar step counters on CPU even when a checkpoint
            # was loaded with map_location pointing to the CUDA device.
            torch.testing.assert_close(value, state[key], rtol=0, atol=0, check_device=False)
        report["checkpoint_reload_exact"] = True
        runner.learn(1)
        report["completed_training_iterations"] = runner.current_learning_iteration
        if runner.current_learning_iteration != args.iterations + 1:
            raise ValueError("Checkpoint resume did not advance the iteration counter")
        policy = runner.get_inference_policy(device=args.device)
        observations = vec.get_observations()[0]
        with torch.inference_mode():
            for _ in range(25):
                observations, _, _, _ = vec.step(policy(observations))
        runner.writer.flush()
        events = EventAccumulator(str(args.output)).Reload()
        tags = events.Tags()["scalars"]
        if not tags or not any(tag.startswith("Loss/") for tag in tags):
            raise ValueError("No training loss metrics were written")
        for tag in tags:
            if not all(math.isfinite(event.value) for event in events.Scalars(tag)):
                raise ValueError(f"Nonfinite training metric: {tag}")
        report["finite_scalar_series"] = len(tags)
        report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        report["checkpoint"] = str(args.output / f"model_{runner.current_learning_iteration}.pt")
    finally:
        if runner.writer is not None:
            runner.writer.close()


def validate(args) -> dict:
    cfg = load_env_cfg(args.task)
    agent = load_instinct_rl_cfg(args.task)
    cfg.seed = agent.seed
    production_envs = cfg.scene.num_envs
    if args.num_envs is not None:
        cfg.scene.num_envs = args.num_envs
    terrain_cfg = cfg.scene.terrain.terrain_generator
    expected_cols = len(terrain_cfg.sub_terrains) if terrain_cfg.curriculum else terrain_cfg.num_cols
    if cfg.scene.num_envs < expected_cols:
        raise ValueError(f"Use at least {expected_cols} test environments to cover every terrain column")
    if args.iterations < 1 or args.steps_per_level < 1:
        raise ValueError("Iterations and steps per level must be positive")
    report = {
        "task": args.task,
        "dependencies": {
            name: version(name) for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "instinct_rl")
        },
        "production_num_envs": production_envs,
        "tested_num_envs": cfg.scene.num_envs,
        "motions": audit_motions(cfg),
        "configured_terrain_shape": [terrain_cfg.num_rows, terrain_cfg.num_cols],
        "terrain_types": list(terrain_cfg.sub_terrains),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    dump_yaml(args.output / "params" / "env.yaml", class_to_dict(cfg))
    dump_yaml(args.output / "params" / "agent.yaml", class_to_dict(agent))
    if args.audit_only:
        report["status"] = "data_audit_passed"
        return report
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("The training preflight requires CUDA; use --audit-only for file validation")
    torch.cuda.set_device(args.device)
    wp.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    report["gpu"] = torch.cuda.get_device_name(args.device)
    report["gpu_total_bytes"] = torch.cuda.get_device_properties(args.device).total_memory
    print("Building the complete curriculum", flush=True)
    env = load_env_cls(args.task)(cfg, device=args.device)
    vec = None
    try:
        report["physics_dt"] = env.sim.mj_model.opt.timestep
        report["decimation"] = cfg.decimation
        if report["physics_dt"] != cfg.sim.mujoco.timestep:
            raise ValueError("Compiled physics timestep differs from the task configuration")
        vec = load_vecenv_cls(args.task)(
            env, policy_group=agent.policy_observation_group, critic_group=agent.critic_observation_group
        )
        report["rollout_observation_bytes"] = (
            cfg.scene.num_envs * agent.num_steps_per_env * (vec.num_obs + vec.num_critic_obs) * 4
        )
        report["production_rollout_observation_bytes"] = (
            production_envs * agent.num_steps_per_env * (vec.num_obs + vec.num_critic_obs) * 4
        )
        report["observation_format"] = vec.get_obs_format()
        report["peak_constraints"] = 0
        report["peak_total_contacts"] = 0
        report["resets"] = 0
        report["physics_world_steps"] = 0
        report["solver_limit_world_steps"] = {"iterations": 0, "line_search": 0}
        original_step = vec.step
        original_physics_step = env.sim.step

        def checked_physics_step():
            original_physics_step()
            report["physics_world_steps"] += env.num_envs
            constraints = int(env.sim.data.nefc.max())
            contacts = int(env.sim.data.nacon.max())
            overflow = env.sim.data.overflow[:]
            # MuJoCo-Warp 3.16 also reports bounded solver iteration stops.
            # Record these separately from capacity/collision data loss.
            limits = mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
            if torch.any(overflow & ~int(limits)):
                raise ValueError(f"MuJoCo Warp collision/capacity overflow: {overflow}")
            for name, flag in (
                ("iterations", mjw.OverflowType.ITERATIONS),
                ("line_search", mjw.OverflowType.LS_ITERATIONS),
            ):
                report["solver_limit_world_steps"][name] += int(torch.count_nonzero(overflow & int(flag)))
            report["peak_constraints"] = max(report["peak_constraints"], constraints)
            report["peak_total_contacts"] = max(report["peak_total_contacts"], contacts)
            if constraints >= cfg.sim.njmax or contacts >= cfg.sim.nconmax * cfg.scene.num_envs:
                raise ValueError("Contact or constraint capacity exhausted")

        env.sim.step = checked_physics_step

        def checked_step(actions):
            _finite(actions, "actions")
            result = original_step(actions)
            _finite(result[:2], "rollout")
            _finite(result[3]["observations"], "observations")
            _finite(env.scene["robot"].data.joint_pos, "robot.joint_pos")
            _finite(env.scene["robot"].data.joint_vel, "robot.joint_vel")
            reference = env.scene["motion_reference"].data
            for name in ("joint_pos", "joint_vel", "base_lin_vel_w", "base_ang_vel_w"):
                _finite(getattr(reference, name), f"reference.{name}")
            report["resets"] += int(result[2].sum())
            return result

        vec.step = checked_step
        _validate_curriculum(env, vec, cfg, args, report)
        # An isolated reset must not alter other robots' state or observation history.
        previous_state = env.scene["robot"].data.joint_pos[1:].clone()
        before = vec.get_observations()[0][1:].clone()
        env.reset(env_ids=torch.zeros(1, dtype=torch.long, device=args.device))
        after = vec.get_observations()[0][1:]
        torch.testing.assert_close(env.scene["robot"].data.joint_pos[1:], previous_state, rtol=0, atol=0)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        report["partial_reset_isolated"] = True
        # Restore normal starting terrain assignments before the learning check.
        terrain = env.scene.terrain
        terrain.terrain_levels.random_(0, cfg.scene.terrain.max_init_terrain_level + 1)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        vec.reset()
        _validate_training(vec, agent, args, report)
        report["status"] = "passed"
        return report
    finally:
        if vec is not None:
            vec.close()
        else:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Instinct-Parkour-Target-Amp-X2-v0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, help="Override only the validation environment count")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--steps-per-level", type=int, default=25)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", type=Path, help="New directory for checkpoints and the JSON report")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("logs/instinct_rl/x2_parkour") / datetime.now().strftime("preflight_%Y%m%d_%H%M%S_%f")
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Use a new output directory so old results cannot be mistaken for this run: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        report = validate(args)
    except Exception as error:
        report = {"status": "failed", "task": args.task, "error": str(error)}
        (args.output / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (args.output / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
