"""Retarget InstinctMJ G1 parkour NPZ motions with an external robot_retargeter checkout.

Run using that tool's environment, for example:
    conda run -n robot_retargeter python src/instinct_mj/scripts/retarget_parkour_motion.py

Source G1 and exported X2 both use torso roots. The external retargeter solves
X2 IK in its pelvis-root convention, followed by an exact torso-root coordinate
conversion. All steps use local MJCF kinematics, with joints mapped by name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RESOURCES = REPO_ROOT / "src/instinct_mj/assets/resources"
G1_XML = RESOURCES / "unitree_g1/xml/g1_29dof_torsobase_popsicle.xml"
X2_SOURCE_XML = RESOURCES / "agibot_x2/x2_ultra_simplified.xml"
X2_TORSO_XML = RESOURCES / "agibot_x2/x2_ultra_simplified_torsobase_fixed_head.xml"
REVERSED_JOINTS = {"waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"}


def prepare_kinematic_model(source: Path, output: Path, robot: str) -> mujoco.MjModel:
    """Keep the exact joint tree, adding only fixed retargeter landmark bodies."""
    root = ET.parse(source).getroot()
    # IK only needs inertias and kinematics, not meshes, sensors or actuators.
    for element in list(root):
        if element.tag in {"asset", "actuator", "sensor", "keyframe"}:
            root.remove(element)
    for parent in root.iter():
        for child in list(parent):
            fixed_head_joint = (
                robot == "agibot_x2"
                and child.tag == "joint"
                and child.get("name")
                in {
                    "head_yaw_joint",
                    "head_pitch_joint",
                }
            )
            if child.tag in {"geom", "light", "camera", "site"} or fixed_head_joint:
                parent.remove(child)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    bodies = {body.get("name"): body for body in root.iter("body")}

    def marker(parent: str, name: str, pos: tuple[float, float, float]) -> None:
        ET.SubElement(bodies[parent], "body", name=name, pos=" ".join(map(str, pos)))

    if robot == "g1":
        marker("pelvis", "hips_sphere", (0.0, 0.0, -0.133165))
        marker("torso_link", "neck_sphere", (0.0, 0.0, 0.247))
        marker("torso_link", "head_sphere", (0.0, 0.0, 0.407))
        heel, toe, sole = -0.05, 0.12, -0.035
    else:
        marker("torso_link", "neck_sphere", (0.00329907, 0.0, 0.241648))
        heel, toe, sole = -0.056, 0.132, -0.0732
    for side in ("left", "right"):
        marker(f"{side}_ankle_roll_link", f"{side}_foot_end_link", (heel, 0.0, sole))
        marker(f"{side}_ankle_roll_link", f"{side}_toe_link", (toe, 0.0, sole))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    original = mujoco.MjModel.from_xml_path(str(source))
    removed_joints = 2 if robot == "agibot_x2" else 0
    if model.nq != original.nq - removed_joints or model.nv != original.nv - removed_joints:
        raise ValueError(f"Unexpected kinematic model dimensions: {output}")
    return model


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]


def convert_qpos_to_torso(
    qpos: np.ndarray,
    pelvis_model: mujoco.MjModel,
    torso_model: mujoco.MjModel,
) -> tuple[np.ndarray, dict[str, float]]:
    """Change coordinates exactly, checking all common body poses frame by frame."""
    target_names = joint_names(torso_model)
    if pelvis_model.body(1).name != "pelvis" or torso_model.body(1).name != "torso_link":
        raise ValueError("Expected pelvis-root source and torso-root target")
    if qpos.ndim != 2 or qpos.shape[1] != pelvis_model.nq or not np.isfinite(qpos).all():
        raise ValueError("Invalid source qpos")
    source = qpos.astype(np.float64, copy=True)
    quat_norms = np.linalg.norm(source[:, 3:7], axis=1, keepdims=True)
    if np.any(quat_norms < 1e-12):
        raise ValueError("Source motion contains a zero root quaternion")
    source[:, 3:7] /= quat_norms
    output = np.zeros((len(source), torso_model.nq), dtype=np.float64)
    for name in target_names:
        target_jid = torso_model.joint(name).id
        source_jid = pelvis_model.joint(name).id
        sign = -1.0 if name in REVERSED_JOINTS else 1.0
        output[:, torso_model.jnt_qposadr[target_jid]] = source[:, pelvis_model.jnt_qposadr[source_jid]] * sign
    before = mujoco.MjData(pelvis_model)
    after = mujoco.MjData(torso_model)
    torso_id = pelvis_model.body("torso_link").id
    source_ids = np.array([pelvis_model.body(torso_model.body(i).name).id for i in range(1, torso_model.nbody)])
    max_pos_error, max_rot_error = 0.0, 0.0
    for frame, positions in enumerate(source):
        before.qpos[:] = positions
        mujoco.mj_forward(pelvis_model, before)
        output[frame, :3] = before.xpos[torso_id]
        output[frame, 3:7] = before.xquat[torso_id]
        after.qpos[:] = output[frame]
        mujoco.mj_forward(torso_model, after)
        max_pos_error = max(max_pos_error, float(np.abs(before.xpos[source_ids] - after.xpos[1:]).max()))
        max_rot_error = max(max_rot_error, float(np.abs(before.xmat[source_ids] - after.xmat[1:]).max()))
    if max_pos_error > 1e-9 or max_rot_error > 1e-9:
        raise ValueError(f"Re-rooting changed body poses: position={max_pos_error}, rotation={max_rot_error}")
    return output, {"max_body_position_error_m": max_pos_error, "max_body_rotation_matrix_error": max_rot_error}


def save_motion(path: Path, qpos: np.ndarray, names: list[str], fps: float, *, overwrite: bool = False) -> None:
    """Write the motion schema atomically for full sequences and clips."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing motion: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        framerate=fps,
        joint_names=np.array(names),
        joint_pos=qpos[:, 7:].astype(np.float32),
        base_pos_w=qpos[:, :3].astype(np.float32),
        base_quat_w=qpos[:, 3:7].astype(np.float32),
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retargeter", type=Path, default=Path.home() / "robot_retargeter")
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "checkpoint/parkour/parkour_motion_reference")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "checkpoint/parkour/x2_parkour_motion_reference")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit each motion for sample validation")
    parser.add_argument("--overwrite", action="store_true", help="Replace previously generated output motions")
    args = parser.parse_args()
    args.retargeter = args.retargeter.resolve()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.input_dir == args.output_dir:
        parser.error("Output must be separate from the source dataset")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least two")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.retargeter / "scripts"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="x2-parkour-") as temporary:
        retarget(args, Path(temporary))


def retarget(args: argparse.Namespace, work: Path) -> None:
    """Run keypoint extraction and IK, then export torso-root references."""
    from robot_retarget import RobotRetarget

    g1_model = prepare_kinematic_model(G1_XML, work / "g1.xml", "g1")
    x2_model = prepare_kinematic_model(X2_SOURCE_XML, work / "x2.xml", "agibot_x2")
    x2_torso_model = mujoco.MjModel.from_xml_path(str(X2_TORSO_XML))
    g1_names, x2_names = joint_names(g1_model), joint_names(x2_torso_model)
    selection_path = args.input_dir / "parkour_motion_without_run.yaml"
    selection = yaml.safe_load(selection_path.read_text())
    split_selection = {"selected_files": [], "motion_weights": []}
    results = []
    for index, relative_file in enumerate(selection["selected_files"]):
        src = args.input_dir / relative_file
        dst = args.output_dir / relative_file
        if dst.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing motion: {dst}")
        with np.load(src, allow_pickle=False) as source:
            source_names = source["joint_names"].tolist()
            count = len(source["joint_pos"])
            if args.max_frames is not None:
                count = min(count, args.max_frames)
            fps = float(source["framerate"])
            qpos = np.zeros((count, g1_model.nq), dtype=np.float64)
            qpos[:, :3] = source["base_pos_w"][:count]
            qpos[:, 3:7] = source["base_quat_w"][:count]
            if count < 2 or not np.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"Motion needs at least two frames and a positive frame rate: {src}")
            quat_norms = np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
            if not np.isfinite(qpos[:, :7]).all() or np.any(quat_norms < 1e-12):
                raise ValueError(f"Invalid root poses: {src}")
            qpos[:, 3:7] /= quat_norms
            qpos[:, 7:] = source["joint_pos"][:count, [source_names.index(name) for name in g1_names]]
        if not np.isfinite(qpos).all():
            raise ValueError(f"Nonfinite joint positions: {src}")
        csv_data = qpos.copy()
        csv_data[:, 3:7] = qpos[:, [4, 5, 6, 3]]  # Retargeter's CSV uses xyzw.
        csv_path = work / f"{index}_g1.csv"
        np.savetxt(csv_path, csv_data, delimiter=",")
        keypoints = work / f"{index}_x2_keypoints.pkl"
        configs = {}
        for robot in ("g1", "agibot_x2"):
            config = yaml.safe_load((args.retargeter / "config/robot" / f"{robot}.yaml").read_text())
            config["robot_xml_path"] = str(work / ("g1.xml" if robot == "g1" else "x2.xml"))
            config["keypoints_path"] = str(keypoints)
            config["verbose"] = False
            config["render_debug"] = False
            # These are source-data motion references, not a bent-knee training pose.
            # Preserve the exact target XML bounds instead of imposing extra offsets.
            config["joints_limit_offset_degrees"] = {}
            path = work / f"{index}_{robot}.yaml"
            # robot_replay's line-based link parser requires inline body-pair lists.
            path.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=None, width=120))
            configs[robot] = (config, path)
        subprocess.run(
            [
                sys.executable,
                str(args.retargeter / "scripts/robot_replay.py"),
                "--no-viewer",
                "--motion-file",
                str(csv_path),
                "--source-robot-config",
                str(configs["g1"][1]),
                "--target-robot-config",
                str(configs["agibot_x2"][1]),
                "--fps",
                str(fps),
                "--output-path",
                str(keypoints),
            ],
            cwd=args.retargeter,
            check=True,
        )
        target_cfg = configs["agibot_x2"][0]
        # The packed source contains concatenated clips. A warm-started root
        # must not spend several output frames chasing a jump between clips.
        restart_frames = set((np.flatnonzero(np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) > 0.2) + 1).tolist())
        restart_frames.add(0)

        class ParkourRetarget(RobotRetarget):
            def update_targets(self, frame_idx):
                if frame_idx in restart_frames:
                    initial = self.configuration.q.copy()
                    hips_idx = self.keypoint_name_to_idx["hips_mean"]
                    initial[:3] = self.keypoints_pos[frame_idx, hips_idx]
                    initial[3:7] = self.keypoints_quat[frame_idx, hips_idx]
                    self.configuration.update(q=initial)
                super().update_targets(frame_idx)

        solver = ParkourRetarget(
            model_path=str(work / "x2.xml"),
            keypoint_path=str(keypoints),
            ik_match_table=target_cfg["ik_match_table"],
            solver="daqp",
            verbose=False,
            render_debug=False,
            joints_limit_offset_degrees={},
            contact_body_names=target_cfg["contact_links"],
            contact_position_cost=target_cfg["contact_pos_fixed_factor"],
        )
        start = time.monotonic()
        solver.retarget()
        output = np.asarray(solver.result_pos, dtype=np.float64)
        if output.shape != (count, x2_model.nq) or not np.isfinite(output).all():
            raise ValueError(f"Invalid retargeted shape/values: {output.shape}")
        output, torso_conversion = convert_qpos_to_torso(output, x2_model, x2_torso_model)
        output[:, 3:7] /= np.linalg.norm(output[:, 3:7], axis=1, keepdims=True)
        # Quaternion sign continuity avoids spurious finite-difference angular velocities.
        for frame in range(1, count):
            if np.dot(output[frame - 1, 3:7], output[frame, 3:7]) < 0.0:
                output[frame, 3:7] *= -1.0
        bounds = x2_torso_model.jnt_range[1:]
        violation = np.maximum(bounds[:, 0] - output[:, 7:], output[:, 7:] - bounds[:, 1])
        max_violation = float(np.maximum(violation, 0.0).max())
        if max_violation > 1e-5:
            raise ValueError(f"X2 joint bounds exceeded by {max_violation} rad")
        save_motion(dst, output, x2_names, fps, overwrite=args.overwrite)
        clip_ranges = []
        edges = sorted(restart_frames) + [count]
        for clip_index, (first, last) in enumerate(zip(edges[:-1], edges[1:])):
            # Preserve source subdirectories: different selections can contain
            # motions with the same basename, whose clips must remain distinct.
            source_relative = Path(relative_file)
            relative_clip = (
                Path("clips") / source_relative.parent / f"{source_relative.stem}_{clip_index:03d}_retargetted.npz"
            )
            save_motion(args.output_dir / relative_clip, output[first:last], x2_names, fps, overwrite=args.overwrite)
            split_selection["selected_files"].append(str(relative_clip))
            source_weight = selection.get("motion_weights", [1.0] * len(selection["selected_files"]))[index]
            split_selection["motion_weights"].append(source_weight * (last - first) / count)
            clip_ranges.append({"file": str(relative_clip), "start": first, "end_exclusive": last})
        results.append({
            "source": str(src),
            "output": str(dst),
            "frames": count,
            "framerate": fps,
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "max_joint_limit_violation_rad": max_violation,
            "torso_root_conversion": torso_conversion,
            "ik_root_restart_frames": sorted(restart_frames),
            "clips": clip_ranges,
            "retarget_seconds": time.monotonic() - start,
        })
        print(json.dumps(results[-1], indent=2), flush=True)
    (args.output_dir / selection_path.name).write_text(yaml.safe_dump(selection, sort_keys=False))
    split_path = args.output_dir / f"{selection_path.stem}_clips.yaml"
    split_path.write_text(yaml.safe_dump(split_selection, sort_keys=False))
    (args.output_dir / "retargeting_report.json").write_text(
        json.dumps(
            {
                "tool": str(args.retargeter),
                "source_model": str(G1_XML),
                "target_model": str(X2_TORSO_XML),
                "ik_target_model": str(X2_SOURCE_XML),
                "ik_root": "pelvis",
                "waist_coordinate_sign_from_ik": -1,
                "source_root": "torso_link",
                "target_root": "torso_link",
                "head_joints": "fixed at zero",
                "ik_joint_limit_offsets": "disabled; exact target XML bounds",
                "motions": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
