# Agibot X2 Ultra

Robot definitions live in `instinct_mj.assets.agibot_x2`, following the
single-module structure of `unitree_g1.py`. Task sensors belong in the parkour
configuration, not the asset module.

```python
import copy

from instinct_mj.assets.agibot_x2 import X2_ACTION_SCALE, get_agibot_x2_robot_cfg

cfg.scene.entities["robot"] = get_agibot_x2_robot_cfg()
cfg.actions["joint_pos"].scale = copy.deepcopy(X2_ACTION_SCALE)
```

## Models and coordinate conventions

- `x2_ultra_simplified.xml` is the unchanged source copied from
  `/home/ubuntu/agibot_x2_urdf/X2_URDF-v1.3.0/`. It uses a pelvis root.
- `x2_ultra_simplified_torsobase_fixed_head.xml` is the model for simulation and
  reference kinematics, exposed as `X2_MJCF_PATH`. It uses `torso_link` as the
  floating root, matching G1. Head yaw/pitch are fixed at zero; head geometry,
  inertias and IMU remain.
- The reversed waist chain negates roll, pitch and yaw coordinates relative to
  the source model. `X2_JOINT_SIGNS_FROM_PELVIS` records the mapping by name.
  Waist yaw bounds become `[-2.2078, 3.43]` rad.
- Native joint order starts with waist roll, pitch, yaw, followed by legs and
  arms. G1 has a different physical waist topology; map by name instead of
  reusing its joint indices or symmetry buffers.

Joint symmetry buffers use G1's `symmetric_augmentation_joint_mapping` and
`symmetric_augmentation_joint_reverse_buf` naming, prefixed with
`X2_29Dof_TorsoBase_`. Reflection across the torso XZ plane swaps left/right
joints, reverses roll/yaw and preserves pitch, knee and elbow coordinates.
These signs differ from the pelvis-to-torso conversion signs. The task config
defines the corresponding reference-link list and permutation; see the
[symmetry notes](../../../tasks/parkour/README.md#symmetry-mappings).

Re-rooting preserves physical body frames, pivots, masses, geometry, contact
settings and inherited joint defaults. The initial torso height is 0.8350706 m,
keeping the original pelvis at 0.68 m and the same physical standing pose.
The pose is an initialization, not a validated standing controller.

## Control settings

PD gains and the nominal joint pose come from `agibot_stiff` and `HOME_KEYFRAME`
in `/home/ubuntu/unitree_rl_mjlab/src/assets/robots/agibot_x2/x2_constants.py`.
Velocity limits are transcribed from the sibling source `x2_ultra.urdf`.
These external checkouts are not runtime dependencies.

Torque limits follow the supplied XML: hips/knees/waist yaw use 120 Nm and wrist
pitch/roll use 4.8 Nm, rather than the gain source's 118 and 2.2 Nm. Action scale
is `0.25 * effort_limit / stiffness`. The asset preserves XML armature (0.03),
friction loss (0.3) and passive damping, uses a 0.9 soft position-limit factor,
and assumes no actuator delay. The parkour factory uses a 1 ms physics timestep
and decimation 20 for a 50 Hz policy.

## Parkour integration

See the [parkour notes](../../../tasks/parkour/README.md#agibot-x2) for camera
configuration, retargeting and side-by-side replay. X2 motions and the robot
must both use the torso-root convention. A complete X2 parkour task is not yet
registered. The environment factory connects sensors, motion references and
symmetry mappings; RL configuration, registration and training validation remain.

The 35 referenced STL meshes and this directory's `LICENSE` come from
`agibot_x2_urdf`. Model and mesh assets retain their upstream Mulan PSL v2 license.
