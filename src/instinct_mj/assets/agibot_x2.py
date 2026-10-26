"""Torso-base Agibot X2 Ultra with fixed head joints and agibot_stiff PD gains."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

X2_RESOURCE_DIR = Path(__file__).parent / "resources" / "agibot_x2"
X2_SOURCE_MJCF_PATH = str(X2_RESOURCE_DIR / "x2_ultra_simplified.xml")
# Match G1's torso-base convention in simulation AND motion-reference kinematics.
X2_MJCF_PATH = str(X2_RESOURCE_DIR / "x2_ultra_simplified_torsobase_fixed_head.xml")
X2_ROOT_BODY = "torso_link"
X2_TORSO_BODY = "torso_link"
X2_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")

# Joint buffers follow the torso-base fixed-head MuJoCo/MJCF native joint order.
# Effort limits come from the supplied XML; velocity limits (rad/s) come
# from the sibling X2_URDF-v1.3.0/x2_ultra.urdf.
X2_JOINT_NAMES: tuple[str, ...] = (
    "waist_roll_joint",
    "waist_pitch_joint",
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
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)

# The reversed waist chain uses positive axes and negated joint coordinates,
# following the G1 torso-base convention. Other physical joint signs are unchanged.
X2_JOINT_SIGNS_FROM_PELVIS: dict[str, float] = {
    name: -1.0 if name.startswith("waist_") else 1.0 for name in X2_JOINT_NAMES
}

# Reflect across the torso XZ plane, following G1's motion-reference buffer API:
# mirrored[..., i] = original[..., mapping[i]] * reverse_buf[i].
# Indices follow X2_JOINT_NAMES, including X2's roll-pitch-yaw waist order.
X2_29Dof_TorsoBase_symmetric_augmentation_joint_mapping = [
    0,  # waist roll
    1,  # waist pitch
    2,  # waist yaw
    9,  # left hip pitch -> right hip pitch
    10,  # left hip roll -> right hip roll
    11,  # left hip yaw -> right hip yaw
    12,  # left knee -> right knee
    13,  # left ankle pitch -> right ankle pitch
    14,  # left ankle roll -> right ankle roll
    3,  # right hip pitch -> left hip pitch
    4,  # right hip roll -> left hip roll
    5,  # right hip yaw -> left hip yaw
    6,  # right knee -> left knee
    7,  # right ankle pitch -> left ankle pitch
    8,  # right ankle roll -> left ankle roll
    22,  # left shoulder pitch -> right shoulder pitch
    23,  # left shoulder roll -> right shoulder roll
    24,  # left shoulder yaw -> right shoulder yaw
    25,  # left elbow -> right elbow
    26,  # left wrist yaw -> right wrist yaw
    27,  # left wrist pitch -> right wrist pitch
    28,  # left wrist roll -> right wrist roll
    15,  # right shoulder pitch -> left shoulder pitch
    16,  # right shoulder roll -> left shoulder roll
    17,  # right shoulder yaw -> left shoulder yaw
    18,  # right elbow -> left elbow
    19,  # right wrist yaw -> left wrist yaw
    20,  # right wrist pitch -> left wrist pitch
    21,  # right wrist roll -> left wrist roll
]

# A reflected rotation axis is an axial vector: roll/yaw reverse, pitch stays.
# Negating waist coordinates during torso re-rooting does not change these signs.
X2_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf = [
    -1,  # waist roll
    1,  # waist pitch
    -1,  # waist yaw
    1,  # left hip pitch
    -1,  # left hip roll
    -1,  # left hip yaw
    1,  # left knee
    1,  # left ankle pitch
    -1,  # left ankle roll
    1,  # right hip pitch
    -1,  # right hip roll
    -1,  # right hip yaw
    1,  # right knee
    1,  # right ankle pitch
    -1,  # right ankle roll
    1,  # left shoulder pitch
    -1,  # left shoulder roll
    -1,  # left shoulder yaw
    1,  # left elbow
    -1,  # left wrist yaw
    1,  # left wrist pitch
    -1,  # left wrist roll
    1,  # right shoulder pitch
    -1,  # right shoulder roll
    -1,  # right shoulder yaw
    1,  # right elbow
    -1,  # right wrist yaw
    1,  # right wrist pitch
    -1,  # right wrist roll
]

X2_JOINT_EFFORT_LIMITS: dict[str, float] = {
    "left_hip_pitch_joint": 120.0,
    "left_hip_roll_joint": 120.0,
    "left_hip_yaw_joint": 120.0,
    "left_knee_joint": 120.0,
    "left_ankle_pitch_joint": 36.0,
    "left_ankle_roll_joint": 24.0,
    "right_hip_pitch_joint": 120.0,
    "right_hip_roll_joint": 120.0,
    "right_hip_yaw_joint": 120.0,
    "right_knee_joint": 120.0,
    "right_ankle_pitch_joint": 36.0,
    "right_ankle_roll_joint": 24.0,
    "waist_yaw_joint": 120.0,
    "waist_pitch_joint": 48.0,
    "waist_roll_joint": 48.0,
    "left_shoulder_pitch_joint": 36.0,
    "left_shoulder_roll_joint": 36.0,
    "left_shoulder_yaw_joint": 24.0,
    "left_elbow_joint": 24.0,
    "left_wrist_yaw_joint": 24.0,
    "left_wrist_pitch_joint": 4.8,
    "left_wrist_roll_joint": 4.8,
    "right_shoulder_pitch_joint": 36.0,
    "right_shoulder_roll_joint": 36.0,
    "right_shoulder_yaw_joint": 24.0,
    "right_elbow_joint": 24.0,
    "right_wrist_yaw_joint": 24.0,
    "right_wrist_pitch_joint": 4.8,
    "right_wrist_roll_joint": 4.8,
}

X2_JOINT_VELOCITY_LIMITS: dict[str, float] = {
    "left_hip_pitch_joint": 11.936,
    "left_hip_roll_joint": 11.936,
    "left_hip_yaw_joint": 11.936,
    "left_knee_joint": 11.936,
    "left_ankle_pitch_joint": 13.087,
    "left_ankle_roll_joint": 15.077,
    "right_hip_pitch_joint": 11.936,
    "right_hip_roll_joint": 11.936,
    "right_hip_yaw_joint": 11.936,
    "right_knee_joint": 11.936,
    "right_ankle_pitch_joint": 13.088,
    "right_ankle_roll_joint": 15.077,
    "waist_yaw_joint": 11.936,
    "waist_pitch_joint": 13.088,
    "waist_roll_joint": 13.088,
    "left_shoulder_pitch_joint": 13.088,
    "left_shoulder_roll_joint": 13.088,
    "left_shoulder_yaw_joint": 15.077,
    "left_elbow_joint": 15.077,
    "left_wrist_yaw_joint": 15.077,
    "left_wrist_pitch_joint": 4.188,
    "left_wrist_roll_joint": 4.188,
    "right_shoulder_pitch_joint": 13.088,
    "right_shoulder_roll_joint": 13.088,
    "right_shoulder_yaw_joint": 15.077,
    "right_elbow_joint": 15.077,
    "right_wrist_yaw_joint": 15.077,
    "right_wrist_pitch_joint": 4.188,
    "right_wrist_roll_joint": 4.188,
}

# agibot_stiff preset from unitree_rl_mjlab/src/assets/robots/agibot_x2/x2_constants.py.
X2_AGIBOT_STIFF_GAINS: dict[str, tuple[float, float]] = {
    ".*_hip_pitch_joint": (120.0, 5.0),
    ".*_hip_roll_joint": (100.0, 4.0),
    ".*_hip_yaw_joint": (100.0, 4.0),
    ".*_knee_joint": (150.0, 5.0),
    ".*_ankle_pitch_joint": (40.0, 2.0),
    ".*_ankle_roll_joint": (40.0, 2.0),
    "waist_yaw_joint": (40.18, 2.56),
    "waist_pitch_joint": (200.0, 2.0),
    "waist_roll_joint": (200.0, 2.0),
    ".*_shoulder_pitch_joint": (50.0, 3.0),
    ".*_shoulder_roll_joint": (50.0, 3.0),
    ".*_shoulder_yaw_joint": (50.0, 3.0),
    ".*_elbow_joint": (50.0, 3.0),
    ".*_wrist_yaw_joint": (20.0, 2.0),
    ".*_wrist_pitch_joint": (20.0, 2.0),
    ".*_wrist_roll_joint": (20.0, 2.0),
}

# Nominal pose from the same x2_constants.py as the agibot_stiff preset.
# At zero waist angles the torso is 0.1550706 m above the 0.68 m pelvis.
# This preserves the original physical pose and foot clearance after re-rooting.
X2_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.8350706),
    joint_pos={
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        ".*_shoulder_pitch_joint": 0.35,
        ".*_elbow_joint": -0.87,
        "left_shoulder_roll_joint": 0.1,
        "right_shoulder_roll_joint": -0.1,
    },
    joint_vel={".*": 0.0},
)


def _joint_gains(name: str) -> tuple[float, float]:
    matches = [gains for pattern, gains in X2_AGIBOT_STIFF_GAINS.items() if re.fullmatch(pattern, name)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one agibot_stiff gain entry for {name}, got {len(matches)}")
    return matches[0]


X2_ACTION_SCALE: dict[str, float] = {
    name: 0.25 * X2_JOINT_EFFORT_LIMITS[name] / _joint_gains(name)[0] for name in X2_JOINT_NAMES
}


def get_agibot_x2_spec() -> mujoco.MjSpec:
    """Load torso-base X2, removing XML motors before mjlab adds PD actuators."""
    spec = mujoco.MjSpec.from_file(X2_MJCF_PATH)
    for actuator in tuple(spec.actuators):
        spec.delete(actuator)
    for body in spec.bodies:
        for light in tuple(body.lights):
            spec.delete(light)
    return spec


def get_agibot_x2_robot_cfg(*, delayed: bool = False) -> EntityCfg:
    """Return a fresh torso-base, 29-joint X2 config with agibot_stiff gains.

    Head yaw and pitch are mechanically fixed at zero in the derived MJCF.
    The waist chain is reversed and its coordinates negated relative to the
    supplied pelvis-base XML. Physical body geometry, inertias, passive damping,
    armature, friction and effort limits are preserved. Gains and the nominal
    physical pose come from unitree_rl_mjlab.
    Action scales use the local XML effort limits, which differ from that repo.
    With ``delayed=True``, apply G1 parkour's per-episode 0--2 physics-step
    command-delay randomization, independently for legs, feet, waist, waist yaw
    and arms. Gains, feedback and physical joint parameters are unchanged.
    """
    return EntityCfg(
        spec_fn=get_agibot_x2_spec,
        init_state=copy.deepcopy(X2_HOME_KEYFRAME),
        sort_actuators=True,
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                BuiltinPdActuatorCfg(
                    target_names_expr=(name,),
                    stiffness=_joint_gains(name)[0],
                    damping=_joint_gains(name)[1],
                    effort_limit=X2_JOINT_EFFORT_LIMITS[name],
                    delay_min_lag=0,
                    delay_max_lag=2 if delayed else 0,
                    delay_update_period=_delay_update_period(name) if delayed else 0,
                    delay_per_env_phase=not delayed,
                )
                for name in X2_JOINT_NAMES
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


def _delay_update_period(name: str) -> int:
    """Match G1's five reset-only command buses without copying its gains."""
    if name == "waist_yaw_joint":
        group = 3
    elif name.startswith("waist_"):
        group = 2
    elif "_ankle_" in name:
        group = 1
    elif "_hip_" in name or "_knee_" in name:
        group = 0
    else:
        group = 4
    return 1_000_000 + group


__all__ = [
    "X2_ACTION_SCALE",
    "X2_AGIBOT_STIFF_GAINS",
    "X2_29Dof_TorsoBase_symmetric_augmentation_joint_mapping",
    "X2_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf",
    "X2_FOOT_BODIES",
    "X2_HOME_KEYFRAME",
    "X2_JOINT_EFFORT_LIMITS",
    "X2_JOINT_NAMES",
    "X2_JOINT_SIGNS_FROM_PELVIS",
    "X2_JOINT_VELOCITY_LIMITS",
    "X2_MJCF_PATH",
    "X2_RESOURCE_DIR",
    "X2_ROOT_BODY",
    "X2_SOURCE_MJCF_PATH",
    "X2_TORSO_BODY",
    "get_agibot_x2_robot_cfg",
    "get_agibot_x2_spec",
]
