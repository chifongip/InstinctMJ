"""Render G1/X2 depth-processing stages from their actual parkour cameras.

Uses the same static obstacle scene and each robot's configured standing pose.
No motion dataset or trained policy is required. Outputs raw arrays and a
paper-style figure; disabled paper augmentations are not synthesized.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from matplotlib.patches import Rectangle
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation

import instinct_mj.tasks  # noqa: F401
from instinct_mj.tasks.registry import load_env_cfg


def _obstacles(spec: mujoco.MjSpec) -> None:
    """Build an asymmetric stair fixture shared by both camera views."""
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], rgba=[0.65, 0.65, 0.65, 1]
    )
    boxes = [
        ("step_1", (0.95, 0, 0.10), (0.25, 0.55, 0.10)),
        ("step_2", (1.45, 0, 0.20), (0.25, 0.55, 0.20)),
        ("step_3", (1.95, 0, 0.30), (0.25, 0.55, 0.30)),
        ("left_marker", (1.1, 0.85, 0.35), (0.16, 0.16, 0.35)),
        ("right_marker", (1.6, -0.85, 0.20), (0.20, 0.20, 0.20)),
    ]
    for name, position, size in boxes:
        spec.worldbody.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=position, size=size, rgba=[0.45, 0.48, 0.5, 1]
        )
    spec.worldbody.add_light(pos=[0, -3, 5], dir=[0, 0, -1])


def _crop_bounds(metadata: dict, *, upright: bool = False) -> tuple[int, int, int, int]:
    """Return retained pixel boundaries in raw or upright image coordinates."""
    height, width = metadata["raw_shape"]
    top, bottom, left, right = metadata["crop_top_bottom_left_right"]
    x0, x1, y0, y1 = left, width - right, top, height - bottom
    if not upright and "upright_image" in metadata["pipeline"]:
        x0, x1, y0, y1 = width - x1, width - x0, height - y1, height - y0
    return x0, x1, y0, y1


def _frustum_corners(metadata: dict, *, cropped: bool) -> np.ndarray:
    """Project image boundaries onto a plane 1 m along the optical axis."""
    height, width = metadata["raw_shape"]
    x0, x1, y0, y1 = _crop_bounds(metadata) if cropped else (0, width, 0, height)
    pixels = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]])
    optical = pixels @ np.linalg.inv(metadata["intrinsics"]).T
    local = optical[:, [2, 0, 1]] * [1, -1, -1]
    rotation = np.empty(9)
    quaternion = np.array(metadata["camera_quaternion_world_wxyz"])
    quaternion /= np.linalg.norm(quaternion)
    mujoco.mju_quat2Mat(rotation, quaternion)
    return local @ rotation.reshape(3, 3).T + metadata["camera_position_world_m"]


def _draw_frustums(scene: mujoco.MjvScene, metadata: dict) -> None:
    """Draw viewer-only wireframes; these are never added to the physics model."""
    if scene.maxgeom - scene.ngeom < 16:
        raise ValueError("Camera frustums require 16 available viewer geometry slots")
    origin = np.array(metadata["camera_position_world_m"])
    for cropped, color in ((False, (0, 0.85, 1, 1)), (True, (1, 0.45, 0, 1))):
        corners = _frustum_corners(metadata, cropped=cropped)
        for index in range(4):
            for start, end in ((origin, corners[index]), (corners[index], corners[(index + 1) % 4])):
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color)
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, start, end)
                scene.ngeom += 1


def _capture(
    robot_name: str, device: str, *, frustum: bool = False, viewer: bool = False
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    cfg = load_env_cfg(f"Instinct-Parkour-Target-Amp-{robot_name}-v0")
    camera_cfg = copy.deepcopy(next(sensor for sensor in cfg.scene.sensors if sensor.name == "camera"))
    scene = Scene(SceneCfg(entities=cfg.scene.entities, sensors=(camera_cfg,), spec_fn=_obstacles), device=device)
    sim = Simulation(num_envs=1, cfg=cfg.sim, device=device, spec=scene.spec, variant_info=scene.collect_variant_info())
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    if scene.sensor_context is not None:
        sim.set_sensor_context(scene.sensor_context)
    robot = scene["robot"]
    robot.write_root_state_to_sim(robot.data.default_root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)
    sim.sense()
    sensor = scene["camera"]
    raw = sensor.data.output["distance_to_image_plane"].clone()
    value = raw.clone()
    stages = {"raw": raw[0, :, :, 0].cpu().numpy()}
    if "upright_image" not in camera_cfg.noise_pipeline:
        stages["upright_image"] = stages["raw"].copy()
    for name, noise in camera_cfg.noise_pipeline.items():
        value = noise.func(value, noise, torch.tensor([0], device=device))
        stages[name] = value[0, :, :, 0].detach().cpu().numpy()
    torch.testing.assert_close(value, sensor.data.output["distance_to_image_plane_noised"], rtol=0, atol=0)
    if value.shape != (1, 18, 32, 1):
        raise ValueError(f"Unexpected policy image shape: {tuple(value.shape)}")
    if not all(np.isfinite(stage).all() for stage in stages.values()):
        raise ValueError("Depth-processing stages contain nonfinite values")
    if value.min() < 0 or value.max() > 1:
        raise ValueError("Normalized policy depth is outside [0, 1]")
    metadata = {
        "task": f"Instinct-Parkour-Target-Amp-{robot_name}-v0",
        "camera_body": camera_cfg.frame.name,
        "camera_position_world_m": sensor.data.pos_w[0].cpu().tolist(),
        "camera_quaternion_world_wxyz": sensor.data.quat_w_world[0].cpu().tolist(),
        "intrinsics": sensor.data.intrinsic_matrices[0].cpu().tolist(),
        "fov_horizontal_vertical_deg": [
            float(np.degrees(2 * np.arctan(size / (2 * focal))))
            for size, focal in zip(
                (camera_cfg.pattern.width, camera_cfg.pattern.height),
                sensor.data.intrinsic_matrices[0].diagonal()[:2].cpu().numpy(),
            )
        ],
        "raw_shape": list(raw.shape[1:3]),
        "policy_shape": list(value.shape[1:3]),
        "crop_top_bottom_left_right": list(camera_cfg.noise_pipeline["crop_and_resize"].crop_region),
        "pipeline": list(camera_cfg.noise_pipeline),
        "crop_resize_shape": camera_cfg.noise_pipeline["crop_and_resize"].resize_shape,
        "blur_kernel_size": camera_cfg.noise_pipeline["gaussian_blur"].kernel_size,
        "blur_sigma": camera_cfg.noise_pipeline["gaussian_blur"].sigma,
        "update_period_s": camera_cfg.update_period,
        "simulation_sensing_period_s": cfg.sim.mujoco.timestep * cfg.decimation,
        "matches_live_sensor_exactly": True,
    }
    # Use the identical compiled scene for an external context view.
    data = mujoco.MjData(sim.mj_model)
    data.qpos[:] = sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(sim.mj_model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.7, 0, 0.7]
    camera.distance, camera.azimuth, camera.elevation = 3.6, 135, -22
    height, width = (600, 800) if frustum else (270, 360)
    sim.mj_model.vis.global_.offheight = max(sim.mj_model.vis.global_.offheight, height)
    sim.mj_model.vis.global_.offwidth = max(sim.mj_model.vis.global_.offwidth, width)
    with mujoco.Renderer(sim.mj_model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        if frustum:
            _draw_frustums(renderer.scene, metadata)
        context = renderer.render().copy()
    if viewer:
        _show_viewer(sim.mj_model, data, camera, metadata)
    return stages, context, metadata


def _show_viewer(model, data, camera, metadata: dict) -> None:
    """Inspect a static scene in MuJoCo's native viewer without stepping physics."""
    import mujoco.viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as window:
        with window.lock():
            window.cam.lookat[:] = camera.lookat
            window.cam.distance, window.cam.azimuth, window.cam.elevation = (
                camera.distance,
                camera.azimuth,
                camera.elevation,
            )
            _draw_frustums(window.user_scn, metadata)
        while window.is_running():
            window.sync()
            time.sleep(1 / 60)


def _figure(captures, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
    fig, axes = plt.subplots(2, 6, figsize=(17, 6.1))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.19, wspace=0.18, hspace=0.55)
    keys = ["raw", "upright_image", "crop_and_resize", "gaussian_blur", "depth_normalization"]
    titles = ["Original Image", "Upright Image", "Crop and Resize", "Gaussian Blur", "Policy Input"]
    for row, (name, stages, context, meta) in enumerate(captures):
        for col, (key, title) in enumerate(zip(keys, titles)):
            ax = axes[row, col]
            image = stages[key]
            ax.imshow(
                image, cmap="gray", vmin=0, vmax=1 if key == "depth_normalization" else 2.5, interpolation="nearest"
            )
            ax.axis("off")
            note = ""
            if key == "upright_image":
                note = "180° rotation" if "upright_image" in meta["pipeline"] else "unchanged"
            if key == "crop_and_resize":
                top, bottom, left, right = meta["crop_top_bottom_left_right"]
                resize = "no resize" if meta["crop_resize_shape"] is None else f"resize to {meta['crop_resize_shape']}"
                note = f"crop T/B/L/R: {top}/{bottom}/{left}/{right}; {resize}"
            if key == "gaussian_blur":
                kernel = meta["blur_kernel_size"]
                note = f"kernel {kernel}, σ = {meta['blur_sigma']}"
            ax.set_title(f"{title}\n{note}", fontsize=10, pad=9)
            ax.text(
                0.5,
                -0.13,
                f"{image.shape[1]} × {image.shape[0]}" + ("  [0, 1]" if key == "depth_normalization" else "  [m]"),
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
            )
            if key in ("raw", "upright_image"):
                # Show the exact retained pixels, expressed in each image frame.
                x0, x1, y0, y1 = _crop_bounds(meta, upright=key == "upright_image")
                ax.add_patch(
                    Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.2)
                )
            if col < 4:
                end = axes[row, col + 1].get_position()
                begin = ax.get_position()
                ax.annotate(
                    "",
                    xy=(end.x0 - 0.003, (begin.y0 + begin.y1) / 2),
                    xytext=(begin.x1 + 0.003, (begin.y0 + begin.y1) / 2),
                    xycoords=fig.transFigure,
                    arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1},
                )
        axes[row, 5].imshow(context)
        axes[row, 5].axis("off")
        axes[row, 5].set_title("Shared Test Scene", fontsize=10, pad=9)
        position = axes[row, 0].get_position()
        fig.text(0.016, (position.y0 + position.y1) / 2, name, fontsize=17, rotation=90, va="center")
    fig.suptitle("Parkour Depth Processing — G1 and Agibot X2", fontsize=17, y=0.99)
    fig.text(
        0.5,
        0.105,
        "Actual task cameras and preprocessing; nominal standing poses in the same static obstacle scene.",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.06,
        "Black = near, white = far. Depth range: 0–2.5 m. Red boxes mark retained pixels. Native pixels are enlarged"
        " without smoothing.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.015,
        "Paper stages not enabled in either task: Gaussian noise, disparity artifacts, OOD perturbations. No"
        " real-camera measurements are shown.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output / "g1_x2_depth_processing.png", dpi=200, facecolor="white")
    fig.savefig(output / "g1_x2_depth_processing.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("videos/depth_comparison"))
    parser.add_argument("--frustum", action="store_true", help="Overlay full FOV (cyan) and policy crop (orange)")
    parser.add_argument(
        "--viewer", choices=("G1", "X2"), help="Open a static, interactive native viewer for this robot"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    captures = []
    for name in ("G1", "X2"):
        stages, context, meta = _capture(name, args.device, frustum=args.frustum, viewer=args.viewer == name)
        np.savez_compressed(args.output_dir / f"{name.lower()}_depth_stages.npz", **stages)
        for key, stage in stages.items():
            plt.imsave(
                args.output_dir / f"{name.lower()}_{key}.png",
                stage,
                cmap="gray",
                vmin=0,
                vmax=1 if key == "depth_normalization" else 2.5,
            )
        captures.append((name, stages, context, meta))
    if args.frustum:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (name, _, context, _) in zip(axes, captures):
            ax.imshow(context)
            ax.set_title(name)
            ax.axis("off")
        fig.suptitle("Cyan: full camera FOV     Orange: retained policy crop")
        fig.text(
            0.5,
            0.02,
            "Frustum display length: 1 m along the optical axis. This is not the sensor range limit.",
            ha="center",
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(args.output_dir / "g1_x2_camera_frustums.png", dpi=200)
        plt.close(fig)
    _figure(captures, args.output_dir)
    (args.output_dir / "metadata.json").write_text(
        json.dumps({name: meta for name, _, _, meta in captures}, indent=2) + "\n"
    )
    print(args.output_dir / "g1_x2_depth_processing.png")


if __name__ == "__main__":
    main()
