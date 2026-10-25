"""Record synchronized G1/X2 motion references as a side-by-side MP4.

This is kinematic data playback, not policy inference or a physics rollout.
Set MUJOCO_GL=egl before launching for headless recording.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mediapy as media
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RESOURCES = REPO_ROOT / "src/instinct_mj/assets/resources"
DATASET = REPO_ROOT / "checkpoint/parkour"
MOTION_NAME = "parkour_motion_without_run_retargetted.npz"


def load_model(xml: Path, width: int, height: int) -> mujoco.MjModel:
    """Add a flat visual floor and fixed lighting without modifying robot assets."""
    spec = mujoco.MjSpec.from_file(str(xml))
    for body in spec.bodies:
        for light in tuple(body.lights):
            spec.delete(light)
    spec.add_texture(
        name="comparison_floor",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=256,
        height=256,
        rgb1=[0.22, 0.25, 0.29],
        rgb2=[0.32, 0.35, 0.39],
    )
    spec.add_material(
        name="comparison_floor",
        textures=["", "comparison_floor"],
        texrepeat=[2, 2],
        texuniform=True,
        reflectance=0.0,
    )
    spec.worldbody.add_geom(
        name="comparison_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material="comparison_floor",
        contype=0,
        conaffinity=0,
        group=0,
    )
    spec.worldbody.add_light(
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[2, -2, 4],
        dir=[-1, 1, -2],
        diffuse=[0.7, 0.7, 0.7],
    )
    spec.worldbody.add_light(
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[-2, 2, 3],
        dir=[1, -1, -2],
        diffuse=[0.35, 0.35, 0.35],
    )
    model = spec.compile()
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.headlight.ambient[:] = [0.25, 0.25, 0.25]
    return model


def load_motion(path: Path, model: mujoco.MjModel) -> tuple[np.ndarray, float]:
    """Map joint columns by name and keep torso-root poses in wxyz convention."""
    with np.load(path, allow_pickle=False) as motion:
        names = motion["joint_names"].tolist()
        qpos = np.zeros((len(motion["joint_pos"]), model.nq))
        qpos[:, :3] = motion["base_pos_w"]
        qpos[:, 3:7] = motion["base_quat_w"]
        for joint_id in range(1, model.njnt):
            name = model.joint(joint_id).name
            qpos[:, model.jnt_qposadr[joint_id]] = motion["joint_pos"][:, names.index(name)]
        fps = float(motion["framerate"])
    if model.body(1).name != "torso_link":
        raise ValueError("Both models must use the torso-root convention")
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid motion: {path}")
    norms = np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError(f"Zero root quaternion: {path}")
    qpos[:, 3:7] /= norms
    return qpos, fps


def text(frame: np.ndarray, message: str, x: int, y: int, scale: float = 0.55) -> None:
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 239, 245), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-motion", type=Path, default=DATASET / "parkour_motion_reference" / MOTION_NAME)
    parser.add_argument("--x2-motion", type=Path, default=DATASET / "x2_parkour_motion_reference" / MOTION_NAME)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "videos/motion_comparison/g1_x2_parkour_comparison.mp4",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive source-frame index")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.width % 2
        or args.height % 2
        or args.width < 640
        or args.height < 360
        or not np.isfinite(args.fps)
        or args.fps <= 0
    ):
        parser.error("Use even width/height of at least 640x360 and a positive FPS")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel_width, panel_height = args.width // 2, args.height - 80
    models = [
        load_model(RESOURCES / "unitree_g1/xml/g1_29dof_torsobase_popsicle.xml", panel_width, panel_height),
        load_model(RESOURCES / "agibot_x2/x2_ultra_simplified_torsobase_fixed_head.xml", panel_width, panel_height),
    ]
    loaded = [load_motion(path, model) for path, model in zip((args.g1_motion, args.x2_motion), models)]
    motions = [item[0] for item in loaded]
    source_fps = loaded[0][1]
    if loaded[1][1] != source_fps or len(motions[0]) != len(motions[1]):
        raise ValueError("Comparison requires equal frame counts and source frame rates")
    end = len(motions[0]) if args.end_frame is None else min(args.end_frame, len(motions[0]))
    if args.start_frame < 0 or args.start_frame >= end:
        raise ValueError("Invalid source frame range")
    count = int(np.ceil((end - args.start_frame) * args.fps / source_fps))
    frame_ids = np.minimum(
        args.start_frame + np.floor(np.arange(count) * source_fps / args.fps).astype(int),
        end - 1,
    )
    cuts = np.flatnonzero(np.linalg.norm(np.diff(motions[0][:, :3], axis=0), axis=1) > 0.2) + 1
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.88]
    camera.distance, camera.azimuth, camera.elevation = 2.8, 35.0, -12.0
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    data = [mujoco.MjData(model) for model in models]
    temporary = args.output.with_name(args.output.stem + ".partial.mp4")
    started = time.monotonic()
    poster_index = int(np.argmin(np.abs(frame_ids - min(args.start_frame + 1000, end - 1))))
    with ExitStack() as stack:
        renderers = [stack.enter_context(mujoco.Renderer(m, height=panel_height, width=panel_width)) for m in models]
        from OpenGL import GL

        renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
        print(f"Renderer: {renderer_name}; recording {count} frames at {args.fps} fps", flush=True)
        writer = stack.enter_context(
            media.VideoWriter(
                temporary,
                shape=(args.height, args.width),
                fps=args.fps,
                crf=21,
                ffmpeg_args=["-preset", "veryfast", "-threads", "4", "-movflags", "+faststart"],
            )
        )
        for output_index, frame_id in enumerate(frame_ids):
            image = np.full((args.height, args.width, 3), (18, 22, 29), dtype=np.uint8)
            for column, (model, state, renderer, motion) in enumerate(zip(models, data, renderers, motions)):
                state.qpos[:] = motion[frame_id]
                # Follow with the camera, preserving motion over the fixed
                # world floor so foot contacts/sliding remain interpretable.
                camera.lookat[:2] = state.qpos[:2]
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera=camera, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                image[40:-40, column * panel_width : (column + 1) * panel_width] = renderer.render()
                label = "G1 source" if column == 0 else "X2 retargeted (torso root)"
                text(image, label, column * panel_width + 16, 27, 0.68)
                text(
                    image,
                    f"recorded torso Z: {motion[frame_id, 2]:.3f} m",
                    column * panel_width + 16,
                    args.height - 14,
                )
            clip = int(np.searchsorted(cuts, frame_id, side="right")) + 1
            text(
                image,
                f"t={frame_id/source_fps:7.2f}s  frame={frame_id:05d}  clip={clip:02d}/{len(cuts)+1}",
                16,
                65,
                0.48,
            )
            text(image, "camera follows XY | recorded world motion | flat floor | kinematic replay", 16, 86, 0.43)
            if any(0 <= frame_id - cut < source_fps / 2 for cut in cuts):
                text(image, "SOURCE CLIP CUT", 16, 107, 0.43)
            writer.add_image(image)
            if output_index in {0, poster_index}:
                cv2.imwrite(str(args.output.with_suffix(".png")), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if output_index % 250 == 0 or output_index == count - 1:
                elapsed = time.monotonic() - started
                print(f"Rendered {output_index+1}/{count}; {elapsed:.1f}s elapsed", flush=True)
    temporary.replace(args.output)
    report = {
        "video": str(args.output.resolve()),
        "renderer": renderer_name,
        "resolution": [args.width, args.height],
        "video_frames": count,
        "video_fps": args.fps,
        "source_fps": source_fps,
        "source_frame_range": [args.start_frame, end],
        "duration_s": count / args.fps,
        "root_body": "torso_link",
        "display_transform": "camera follows root XY; recorded qpos unchanged",
        "camera": {"azimuth": 35, "elevation": -12, "distance": 2.8},
        "replay": "kinematic; no physics stepping or policy",
        "source_clip_cuts": cuts.tolist(),
        "motions": [
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (args.g1_motion, args.x2_motion)
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Saved {args.output}; duration {count/args.fps:.2f}s; render time {time.monotonic()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
