# Parkour Task

The registered parkour tasks target G1 and Agibot X2. The [X2 asset notes](../../assets/resources/agibot_x2/README.md)
and [X2 configuration guide](#agibot-x2) cover the robot, camera and reference
motions, environment factory and RL configuration for X2.

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

### Environment factory

`config/agibot_x2/x2_parkour_target_amp_cfg.py` follows G1's scene/environment
classes and base/final factories, using shared parkour MDP and terrain components:

```python
from instinct_mj.tasks.parkour.config.agibot_x2.x2_parkour_target_amp_cfg import (
    instinct_x2_parkour_amp_final_cfg,
)

cfg = instinct_x2_parkour_amp_final_cfg(play=False)
# Optional: dataset_dir="/path/to/x2_parkour_motion_reference"
```

The factory uses X2's fixed-head torso-root model, nominal pose and `agibot_stiff`
gains. All 29 movable joints are policy-controlled, matching G1 parkour's
full-body action layout. Physics matches G1: a 5 ms timestep, four substeps per
50 Hz policy/AMP step and per-episode command delays of 0–2 physics steps
(0–10 ms), grouped into the same five actuator buses.

Motion loading defaults to `checkpoint/parkour/x2_parkour_motion_reference/`
relative to this checkout, selecting only the 50 discontinuity-free clips.
Joint and link symmetry buffers are connected to the motion-reference manager.
The policy, critic, AMP policy and AMP reference groups follow G1's layout.
The camera renders 64×36 depth frames. Task preprocessing rotates the inverted
raw image by 180 degrees, then applies G1's crop (top 18 pixels and 16 pixels
from each side), blur and normalization. Policy and critic receive eight
32×18 frames, with the same encoder input and network parameter shapes as G1.
The 37-frame history samples every five simulation camera frames at 50 Hz,
matching G1's approximately 0.7-second observation span. The hardware capture
mode is 480×270 at 60 Hz, downsampled to 64×36 before cropping.

Contact sensors use X2's authored collisions. Foot volume bounds include the
capsules' 0.015 m radius, and the foot-height reward uses a 0.0732 m sole offset.
Other reward weights, commands and curricula start from G1's settings and still
need X2 training evaluation. Holding the nominal joint targets alone is not a
balance controller; sustained standing still needs feedback control.

### Training and playback

X2 follows G1's `rl_cfg.py` and registration layout, with its own experiment
name, `x2_parkour`. The initial runner uses the same depth encoder, four-expert
MoE policy, WasabiPPO discriminator and PPO hyperparameters as G1.

```bash
uv run instinct-train Instinct-Parkour-Target-Amp-X2-v0 --device cuda:0
uv run instinct-play Instinct-Parkour-Target-Amp-X2-Play-v0 --load-run <run_name>
```

Both registered IDs provide train/play factories. Training requires CUDA.
The X2 constraint buffer uses 2048 rows: the initial stair-scene check exceeded
G1's 700-row budget before policy construction. Long training, reward tuning
and evaluation across the full terrain curriculum remain necessary.

### Comparison with G1

The train/play factories have the same observation groups, observation history lengths,
reward weights, commands, events, terminations and curriculum thresholds. PPO,
AMP, depth-encoder and four-expert MoE hyperparameters match exactly; only the
RL experiment name changes from `g1_parkour` to `x2_parkour`. Both control 29
joints at 50 Hz, use 2048 training environments and have 20-second episodes.

| Setting | G1 parkour (default shoe model) | X2 parkour |
| --- | --- | --- |
| Physics timestep / decimation | 5 ms / 4 | 5 ms / 4 |
| Actuation | G1 gains; 0–2 physics-step lag | `agibot_stiff`; same delay scheme |
| Nominal torso height | 0.9 m | 0.8350706 m, preserving 0.68 m pelvis height |
| Joint limits / action scales | G1-specific | X2 XML/URDF limits; scale = 0.25 × effort / stiffness |
| Constraint rows per environment | 700 | 2048 |
| Foot sole offset | 0.058 m | 0.0732 m; volume points follow X2 capsules |
| Camera parent / ray orientation | Torso / yaw only | Fixed head / full parent orientation |
| Depth FOV (horizontal × vertical) | 89.51° × 58.29° | 90° × 60° |
| Raw → policy depth image (width × height) | 64×36 → 32×18 after cropping | Same; rotate raw image 180° before cropping |
| Camera refresh in simulation | Each policy sensing step (50 Hz) | Same; hardware capture mode is 60 Hz |
| Eight depth frames | Five-sample spacing, about 0.70 s | Same |
| Minimum ray-hit rejection distance | 0.1 m | 0.1 m |
| Reference wrist endpoints | Wrist-yaw links | Wrist-roll links (X2 terminal wrists) |
| Motion selection | G1 motion YAML | Retargeted X2 clip YAML, torso root |

Remaining differences follow X2's geometry, joint conventions, requested gains
and camera specifications. The camera's raw mounting stays unchanged; only
policy preprocessing rotates the image. The 0.1 m ray-hit filter matches G1
and is not a calibrated stereo-camera minimum-depth model.

The policy architecture now matches G1, including inferred layer dimensions.
G1 weights still require robot-specific adaptation because the joint meanings,
order and action scales differ. Earlier X2 smoke-test checkpoints using 80×50
policy images have incompatible encoder dimensions and must be retrained.

### Full-training readiness

Production uses 2048 environments, 5 ms physics, a 50 Hz policy, eight 32×18
depth frames, 24 rollout steps and four PPO minibatches. The curriculum has
100 tiles: ten difficulty rows and one column per terrain type. mjlab ignores
`num_cols=20` in curriculum mode; spawn proportions and metadata follow the
actual columns.

On the server, install the pinned mjlab revision using `uv.lock`, and copy
`checkpoint/parkour/x2_parkour_motion_reference/`, including the clip YAML and
`clips/`. Motion data is Git-ignored. The updated mjlab requires
MuJoCo/MuJoCo-Warp 3.16 and Warp 1.15 or newer.

```bash
uv sync --locked
uv run instinct-list-envs --keyword X2
uv run python -m instinct_mj.scripts.validate_parkour --audit-only
uv run python -m instinct_mj.scripts.validate_parkour --device cuda:0
uv run instinct-train Instinct-Parkour-Target-Amp-X2-v0 --device cuda:0
```

The preflight audits motion files, checks all terrain tiles and physics
substeps, verifies curriculum and partial resets, runs two training iterations,
restores model/optimizer state, resumes one iteration and tests inference.
Checkpoints, parameter snapshots, dependency versions and `readiness.json` go
into a new directory under `logs/instinct_rl/x2_parkour/`. Solver iteration-limit
flags are reported separately from collision/capacity overflows.
Use `--num-envs 32` for local validation; registered task defaults stay unchanged.
Run the production-batch preflight on the server before full training.

Local checks passed with 32 environments on mjlab revision `8bb2bc9e1`,
MuJoCo/MuJoCo-Warp 3.16 and Warp 1.18, including all 100 tiles, checkpoint
resume and finite training metrics. Shared motion parsing also matches both
robots' MuJoCo body frames, including off-origin joint pivots. Full training
and learned parkour performance remain unverified.

The motion audit found 68 of 18,932 transitions (0.36%) above URDF velocity
limits, with a maximum ratio of 1.81. These remain AMP style references rather
than direct joint-position tracking commands.

To resume a saved run with its optimizer state:

```bash
uv run instinct-train Instinct-Parkour-Target-Amp-X2-v0 --device cuda:0 \
  --agent.resume --agent.load-run /absolute/path/to/run
```

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
image rotation in deployment; the helper preserves the raw mounting orientation
and the parkour task rotates the policy image into an upright convention.

The selected hardware depth mode is **480×270 at 60 Hz**, with a **90°×60°**
FOV, following the paper's capture rate and resolution. Orbbec's
[Gemini 330 datasheet, page 24](https://www.orbbec.com.cn/uploads/20250422/fa9480bb72bd788196829d1711a0e08d.pdf#page=24)
lists the stream mode; [page 36](https://www.orbbec.com.cn/uploads/20250422/fa9480bb72bd788196829d1711a0e08d.pdf#page=36)
lists its FOV. The 90°×65° specification belongs to the 16:10 modes.

Like G1, simulation raycasts directly at 64×36 and senses once per 50 Hz
policy step; it does not render all 60 hardware frames per second. The task
keeps five-sample history spacing, an approximately 0.7-second history span,
and G1's 32×18 cropped policy input. Camera mounting and the 2.5 m clipping
limit are unchanged. Actual robot deployment must select the 480×270/60 Hz
stream explicitly; these settings configure simulation, not the robot driver.

To visualize both task cameras and their processing stages in the same static
obstacle scene, run:

```bash
uv run python -m instinct_mj.scripts.visualize_parkour_depth
```

The figure, individual stages, raw arrays and camera metadata are saved under
`videos/depth_comparison/`. The script checks that its final images exactly
match the live sensor pipeline. Paper augmentations that are disabled in the
tasks are identified in the caption.

For FOV verification, add `--frustum` to draw the full camera frustum in cyan
and the retained policy crop in orange. This also saves
`g1_x2_camera_frustums.png`. The viewer-only wireframes extend 1 m along the
optical axis and do not affect sensing or physics.

On a machine with a desktop display, open an interactive static X2 scene with:

```bash
MUJOCO_GL=glfw uv run python -m instinct_mj.scripts.visualize_parkour_depth --frustum --viewer X2
```

Use `--viewer G1` for the same inspection on G1. Close the viewer to finish
writing the comparison figure. Training and playback settings are unchanged.

### Motion references

Run the adapter in the installed retargeter environment:

```bash
conda run --no-capture-output -n robot_retargeter python \
  src/instinct_mj/scripts/retarget_parkour_motion.py
```

Defaults are `~/robot_retargeter` for the tool, `checkpoint/parkour/parkour_motion_reference/`
for the source, and `checkpoint/parkour/x2_parkour_motion_reference/` for output.
Use `--retargeter`, `--input-dir`, `--output-dir`, `--max-frames` and `--overwrite`
to override them. Exported clip paths retain source subdirectories so motions
with equal basenames remain distinct. Intermediate CSVs, keypoints and IK models
are temporary.
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
semantic ordering and using X2's terminal wrist-roll links. The motion-reference
manager uses these as `link_of_interests` and `symmetric_augmentation_link_mapping`,
with the two joint augmentation fields supplied by the asset buffers. Both
registered tasks use these settings.

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
