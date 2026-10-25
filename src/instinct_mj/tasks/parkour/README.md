# Parkour Task

The registered parkour tasks currently target G1. The [X2 asset notes](../../assets/resources/agibot_x2/README.md)
and [X2 configuration guide](#agibot-x2) cover the robot, camera and reference
motions prepared for an X2 task, which is not yet registered.

## Basic Usage Guidelines

### Parkour Task

**Task IDs:**
- `Instinct-Parkour-Target-Amp-G1-v0` (train)
- `Instinct-Parkour-Target-Amp-G1-Play-v0` (play)

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set the `path` and `filtered_motion_selection_filepath` in `AmassMotionCfg` to the reference motion you want to use.

   ```python
   path: str = os.path.expanduser("~/your/path/to/parkour_motion_reference")
   filtered_motion_selection_filepath: str | None = os.path.join(
       path,
       "parkour_motion_without_run.yaml",
   )
   ```

   Keep the selected motion `.npz` files and the selection `.yaml` aligned with the same dataset root unless you intentionally split them.

2. Train the policy:
```bash
instinct-train Instinct-Parkour-Target-Amp-G1-v0
```

3. Play trained policy (`--load-run` must be provided, absolute path is recommended, or use `--agent random` to visualize an untrained policy):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name>
```

4. Export trained policy (`--load-run` must be provided, absolute path is recommended):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

5. Use the exported ONNX policy for play:

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --use-onnx
```

## Onboard Joint Order

If you deploy the exported ONNX policy in `instinct_onboard`, make sure `instinct_onboard/instinct_onboard/robot_cfgs.py` uses the `InstinctMJ` MuJoCo joint order instead of the older InstinctLab-style shoulder-first order.

Current `instinct_onboard` `sim_joint_names` order is shoulder-first:

```python
[
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
```

`InstinctMJ` G1 MJCF order is:

```python
[
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
```

When switching `instinct_onboard` to the `InstinctMJ` order, update the following in `instinct_onboard/instinct_onboard/robot_cfgs.py` together:

1. Replace `sim_joint_names` with the `InstinctMJ` order above.
2. Recompute `joint_map` from `real_joint_names` using the same order:

```python
joint_map = [real_joint_names.index(name) for name in sim_joint_names]
```

3. Reorder every array that is documented as "in simulation order" to match the new `sim_joint_names`, especially `joint_signs`, `joint_limits_high`, `joint_limits_low`, and `torque_limits`.

You can safely generate those arrays by name instead of editing indices by hand:

```python
old_sim_joint_names = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

def reorder_from_old(values, sim_joint_names):
    by_name = {name: values[i] for i, name in enumerate(old_sim_joint_names)}
    return [by_name[name] for name in sim_joint_names]
```

This matters because `instinct_onboard/instinct_onboard/agents/base.py` and `instinct_onboard/instinct_onboard/agents/parkour_agent.py` both parse observation and action tensors by iterating `sim_joint_names`. If the onboard order stays on the old InstinctLab convention while the exported `InstinctMJ` config/ONNX uses MJ order, joint observations, action offsets, action scales, and zero-action masks will all be misaligned.

## Common Options

- `--num-envs`: Number of parallel environments (default varies by task)
- `--load-run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--export-onnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--use-onnx`: Use the ONNX model for inference during playing

## Agibot X2

### Depth camera

Like G1, X2 configures its camera at task level, in
`config/agibot_x2/x2_parkour_target_amp_cfg.py`:

```python
from instinct_mj.tasks.parkour.config.agibot_x2.x2_parkour_target_amp_cfg import (
    get_agibot_x2_depth_camera_cfg,
)

cfg.scene.sensors = (*cfg.scene.sensors, get_agibot_x2_depth_camera_cfg())
```

The `rgbd_head_front` URDF mount is relative to `head_pitch_link`: position
`(0.05761, -0.011183, -0.04837)` m and RPY `(2.2689, 0, 1.5708)` rad.
Using ROS optical axes (+Z forward, +X right, +Y down), the raw image is upside
down and the camera points approximately 40 degrees downward. Match any hardware
image rotation in training; the factory preserves the raw mounting orientation.

The [Gemini335 specifications](https://x2-aimdk.agibot.com/en/latest/about_agibot_X2/sensor_fov.html#rgbd)
give a depth FOV of 90 x 65 degrees and resolution up to 1280x800. The factory
sets both FOV axes explicitly, defaults to 80x50, and follows the head's full
pose. Its 30 Hz update period matches the published
[depth-image topic rate](https://x2-aimdk.agibot.com/en/latest/Interface/hal/sensor.html).
The 2.5 m clipping limit is a task default. Noise, preprocessing and history are
left to the task; measured calibration and driver rotation need hardware checks.

### Motion references

Run the adapter in the installed retargeter environment:

```bash
conda run --no-capture-output -n robot_retargeter python \
  src/instinct_mj/scripts/retarget_parkour_motion.py
```

Defaults are `~/robot_retargeter` for the tool, `checkpoint/parkour/parkour_motion_reference/`
for the source, and `checkpoint/parkour/x2_parkour_motion_reference/` for output.
Use `--retargeter`, `--input-dir`, `--output-dir`, `--max-frames` and `--overwrite`
to override them. Intermediate CSVs, keypoints and IK models are temporary.
The original dataset is preserved.

The adapter maps source joints by name, extracts/scales targets using the tool,
and runs contact-aware Mink/DAQP IK against X2's pelvis-root kinematics with the
head fixed. It then exports torso-root poses and negated waist coordinates,
checking world-space body-pose equivalence against the training model. Extra
retargeter knee/elbow limit offsets are disabled; XML joint bounds are enforced.
Output NPZ fields are `framerate`, `joint_names`, `joint_pos`, `base_pos_w` and
`base_quat_w` (wxyz), with joints in X2's native torso-base order.

The supplied dataset contains 18,982 frames at 50 Hz. Its 49 root jumps above
0.2 m/frame divide it into 50 clips. Select **`parkour_motion_without_run_clips.yaml`**
for training so velocity estimates and AMP histories do not cross those cuts.
Length-proportional `motion_weights` preserve the source time distribution.
The full sequence and its original selection YAML are also retained. Always
use a selection YAML to avoid loading both full and split copies.

`retargeting_report.json` records provenance and conversion checks. The current
dataset's `validation_report.json` contains additional FK diagnostics. Retargeting
is kinematic; visual/contact review and dynamic feasibility checks remain necessary.

### Symmetry mappings

`assets/agibot_x2.py` exports G1-style joint permutation and sign buffers:
`X2_29Dof_TorsoBase_symmetric_augmentation_joint_mapping` and
`X2_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf`. They mirror
positions or velocities as `mirrored[..., i] = original[..., mapping[i]] * signs[i]`
in X2's native joint order. Waist roll/pitch/yaw signs are `[-1, +1, -1]`.

The X2 task config defines `X2_PARKOUR_REFERENCE_LINK_NAMES` and
`X2_PARKOUR_SYMMETRIC_LINK_MAPPING` for 14 reference links, following G1's
semantic ordering and using X2's terminal wrist-roll links. When adding the
motion-reference manager, copy these into `link_of_interests` and
`symmetric_augmentation_link_mapping`, and set the two joint augmentation fields
from the asset buffers. The mappings are prepared; an X2 environment is not yet registered.

The model has small authored left/right differences, so reflected FK poses are
approximate, especially in the arms. Waist yaw bounds are also asymmetric;
check mirrored targets against joint limits when using new datasets. All current
18,982 reference frames pass mirrored joint-limit checks.

### Comparison video

With `ffmpeg` and offscreen EGL rendering available:

```bash
MUJOCO_GL=egl uv run --no-sync python src/instinct_mj/scripts/compare_parkour_motions.py
```

This writes an MP4, PNG preview and recording metadata under
`videos/motion_comparison/`. The default is 1280x720 at 25 fps, showing every other
50 Hz source frame at real-time speed; `--fps 50` includes every frame. Both panels
share frame indices and camera angles. Cameras follow recorded XY trajectories
over a fixed floor, preserving robot poses so foot motion remains interpretable.
Labels identify timestamps, source cuts and torso height. This is motion-data
playback; contact and policy stability require separate rollout validation.

For a short recording, add `--start-frame 0 --end-frame 500` and a different
`--output` path. `--end-frame` is exclusive. Other options include `--g1-motion`,
`--x2-motion`, `--width`, `--height` and `--overwrite`. GPU rendering is recommended
for the full meshes.
