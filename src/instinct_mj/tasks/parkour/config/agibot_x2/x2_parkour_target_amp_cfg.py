"""Agibot X2 parkour AMP task config factories.

Config is built via factory functions that return a fully-built
``X2ParkourAmpEnvCfg``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    PinholeCameraPatternCfg,
    RayCastSensorCfg,
)
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.envs.mdp as envs_mdp
import instinct_mj.tasks.parkour.mdp as parkour_mdp
from instinct_mj.assets.agibot_x2 import (
    X2_ACTION_SCALE,
    X2_JOINT_VELOCITY_LIMITS,
    X2_MJCF_PATH,
    X2_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    X2_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    get_agibot_x2_robot_cfg,
)
from instinct_mj.envs.manager_based_rl_env_cfg import InstinctRlEnvCfg
from instinct_mj.managers import MultiRewardCfg
from instinct_mj.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinct_mj.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg
from instinct_mj.motion_reference.utils import motion_interpolate_bilinear
from instinct_mj.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg
from instinct_mj.sensors.volume_points import Grid3dPointsGeneratorCfg, VolumePointsCfg
from instinct_mj.tasks.parkour.config.parkour_env_cfg import (
    ROUGH_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG_PLAY,
    _edit_parkour_scene_spec,
)
from instinct_mj.tasks.parkour.mdp.commands import PoseVelocityCommandCfg
from instinct_mj.terrains.terrain_importer_cfg import TerrainImporterCfg as InstinctTerrainImporterCfg
from instinct_mj.terrains.virtual_obstacle.edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg
from instinct_mj.utils.noise import DepthNormalizationCfg, GaussianBlurNoiseCfg

# Repository-local default; callers can supply a different dataset directory.
_PARKOUR_DATASET_DIR = Path(__file__).resolve().parents[6] / "checkpoint/parkour/x2_parkour_motion_reference"

# Match G1's semantic reference-link ordering, using X2's terminal wrist links.
# The link permutation is indexed by this list, not MuJoCo body IDs.
X2_PARKOUR_REFERENCE_LINK_NAMES = [
    "pelvis",
    "torso_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
X2_PARKOUR_SYMMETRIC_LINK_MAPPING = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]

# rgbd_head_front fixed joint in X2_URDF-v1.3.0/x2_ultra.urdf.
# RPY=(2.2689, 0, 1.5708), ROS optical axes: +Z forward, +X right, +Y down.
# Preserve the raw upside-down mounting; do not apply an extra image flip.
X2_DEPTH_CAMERA_BODY = "head_pitch_link"
X2_DEPTH_CAMERA_POS = (0.05761, -0.011183, -0.04837)
X2_DEPTH_CAMERA_QUAT_ROS = (0.29884467067118387, 0.6408510171708242, 0.6408533711523742, 0.2988457683909694)
# Gemini335 depth specifications (not the RGB channel's 94 x 68 degree FOV).
X2_DEPTH_CAMERA_FOV = (90.0, 65.0)
X2_DEPTH_CAMERA_NATIVE_RESOLUTION = (1280, 800)
# AimDK's published depth_image topic rate; hardware supports up to 60 fps.
X2_DEPTH_CAMERA_UPDATE_PERIOD = 1.0 / 30.0


def get_agibot_x2_depth_camera_cfg(
    *,
    name: str = "camera",
    width: int = 80,
    height: int = 50,
    max_distance: float = 2.5,
) -> NoisyGroupedRayCasterCameraCfg:
    """Build raw X2 depth sensing with the URDF pose and nominal depth FOV.

    Add the result to ``cfg.scene.sensors``. The default image is downsampled
    16 times from 1280x800, retaining both FOV axes and the upside-down image.
    ``max_distance`` is a task raycast/clipping limit, not a hardware spec.
    Noise, cropping, normalization and observation history are task settings.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Depth camera width and height must be positive")
    if not math.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("Depth camera max_distance must be finite and positive")
    return NoisyGroupedRayCasterCameraCfg(
        name=name,
        frame=ObjRef(type="body", name=X2_DEPTH_CAMERA_BODY, entity="robot"),
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=X2_DEPTH_CAMERA_POS,
            rot=X2_DEPTH_CAMERA_QUAT_ROS,
            convention="ros",
        ),
        ray_alignment="base",
        pattern=PinholeCameraPatternCfg(width=width, height=height, fovy=X2_DEPTH_CAMERA_FOV[1]),
        focal_length=1.0,
        horizontal_aperture=2.0 * math.tan(math.radians(X2_DEPTH_CAMERA_FOV[0]) / 2.0),
        vertical_aperture=2.0 * math.tan(math.radians(X2_DEPTH_CAMERA_FOV[1]) / 2.0),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        update_period=X2_DEPTH_CAMERA_UPDATE_PERIOD,
        max_distance=max_distance,
    )


# ---------------------------------------------------------------------------
# Motion reference configs
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class AmassMotionCfg(AmassMotionCfgBase):
    """Retargeted X2 clips in the native torso-root joint convention."""

    path: str = str(_PARKOUR_DATASET_DIR)
    retargetting_func: object | None = None
    filtered_motion_selection_filepath: str | None = str(_PARKOUR_DATASET_DIR / "parkour_motion_without_run_clips.yaml")
    motion_start_from_middle_range: list[float] = field(default_factory=lambda: [0.0, 0.9])
    motion_start_height_offset: float = 0.0
    ensure_link_below_zero_ground: bool = False
    buffer_device: str = "output_device"
    motion_interpolate_func: object = field(default_factory=lambda: motion_interpolate_bilinear)
    velocity_estimation_method: str = "frontward"


motion_reference_cfg = MotionReferenceManagerCfg(
    name="motion_reference",
    entity_name="robot",
    robot_model_path=X2_MJCF_PATH,
    link_of_interests=list(X2_PARKOUR_REFERENCE_LINK_NAMES),
    symmetric_augmentation_link_mapping=list(X2_PARKOUR_SYMMETRIC_LINK_MAPPING),
    symmetric_augmentation_joint_mapping=list(X2_29Dof_TorsoBase_symmetric_augmentation_joint_mapping),
    symmetric_augmentation_joint_reverse_buf=list(X2_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf),
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={"run_walk": AmassMotionCfg()},
    mp_split_method="Even",
)


@dataclass(kw_only=True)
class X2ParkourSceneCfg(SceneCfg):
    """Scene configuration for the X2 Parkour task."""


@dataclass(kw_only=True)
class X2ParkourAmpEnvCfg(InstinctRlEnvCfg):
    """Dictionary-manager environment configuration for X2 Parkour AMP."""

    scene: X2ParkourSceneCfg = field(default_factory=X2ParkourSceneCfg)
    decimation: int = 20
    observations: dict = field(default_factory=dict)
    actions: dict = field(default_factory=dict)
    rewards: dict = field(default_factory=lambda: MultiRewardCfg({"rewards": {}}))
    terminations: dict = field(default_factory=dict)
    commands: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    curriculum: dict = field(default_factory=dict)
    monitors: dict = field(default_factory=dict)


def instinct_x2_parkour_amp_env_cfg(
    *,
    play: bool = False,
    dataset_dir: str | Path | None = None,
) -> X2ParkourAmpEnvCfg:
    """Build the base X2 parkour AMP environment configuration.

    Args:
      play: If True, apply play-mode overrides (fewer envs, relaxed
        termination, etc.).
      dataset_dir: Directory containing retargeted clips and their selection YAML.

    Returns:
      A ``X2ParkourAmpEnvCfg`` instance with parkour settings applied.
    """
    tracking_cfg = make_tracking_env_cfg()
    tracking_scene = tracking_cfg.scene
    cfg = X2ParkourAmpEnvCfg(
        decimation=tracking_cfg.decimation,
        scene=X2ParkourSceneCfg(
            num_envs=tracking_scene.num_envs,
            env_spacing=tracking_scene.env_spacing,
            terrain=tracking_scene.terrain,
            entities={"robot": get_agibot_x2_robot_cfg()},
            sensors=tracking_scene.sensors,
            extent=tracking_scene.extent,
            spec_fn=tracking_scene.spec_fn,
        ),
        observations=tracking_cfg.observations,
        actions=tracking_cfg.actions,
        events=tracking_cfg.events,
        seed=tracking_cfg.seed,
        sim=tracking_cfg.sim,
        viewer=tracking_cfg.viewer,
        episode_length_s=tracking_cfg.episode_length_s,
        rewards=MultiRewardCfg({"rewards": tracking_cfg.rewards}),
        terminations=tracking_cfg.terminations,
        commands=tracking_cfg.commands,
        curriculum=tracking_cfg.curriculum,
        metrics=tracking_cfg.metrics,
        recorders=tracking_cfg.recorders,
        is_finite_horizon=tracking_cfg.is_finite_horizon,
        auto_reset=tracking_cfg.auto_reset,
        scale_rewards_by_dt=tracking_cfg.scale_rewards_by_dt,
        monitors={},
    )
    cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    cfg.viewer.entity_name = None
    cfg.viewer.body_name = None

    # Basic settings
    cfg.scene.num_envs = 2048
    cfg.scene.env_spacing = 2.5
    cfg.episode_length_s = 20.0
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 700
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.mujoco.iterations = 10
    cfg.sim.mujoco.ls_iterations = 20
    cfg.sim.mujoco.ccd_iterations = 128
    # Retain the XML's 1 ms physics step for X2's stiff PD gains; policy/AMP stay at 50 Hz.
    cfg.sim.mujoco.timestep = 0.001
    cfg.decimation = 20
    joint_pos_action: JointPositionActionCfg = cfg.actions["joint_pos"]
    joint_pos_action.scale = copy.deepcopy(X2_ACTION_SCALE)
    # Terrain
    terrain_gen = copy.deepcopy(ROUGH_TERRAINS_CFG_PLAY if play else ROUGH_TERRAINS_CFG)
    edge_obstacle_cfg = GreedyconcatEdgeCylinderCfg(
        cylinder_radius=0.05,
        min_points=2,
        component_workers=0,
        merge_collinear_gap=0.09,
        merge_collinear_angle_threshold=30.0,
        merge_collinear_line_distance=0.04,
    )
    cfg.scene.terrain = InstinctTerrainImporterCfg(
        terrain_type="hacked_generator",
        terrain_generator=copy.deepcopy(terrain_gen),
        max_init_terrain_level=5,
        virtual_obstacle_source="mesh",
        virtual_obstacle_hfield_height_threshold=0.04,
        collision_debug_vis=True,
        collision_debug_rgba=(0.62, 0.2, 0.9, 0.35),
        virtual_obstacles={
            "edges": edge_obstacle_cfg,
        },
    )
    # Scene visual style
    cfg.scene.spec_fn = _edit_parkour_scene_spec
    # Scene sensors
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(
                mode="body",
                pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
                entity="robot",
            ),
            fields=("found", "force"),
            reduce="netforce",
            track_air_time=True,
            history_length=3,
        ),
        ContactSensorCfg(
            name="torso_contact_forces",
            primary=ContactMatch(mode="body", pattern="torso_link", entity="robot"),
            secondary=None,
            fields=("force",),
            reduce="netforce",
            track_air_time=False,
            history_length=3,
        ),
        ContactSensorCfg(
            name="undesired_contact_forces",
            primary=ContactMatch(
                mode="body",
                pattern=".*",
                entity="robot",
                exclude=("left_ankle_roll_link", "right_ankle_roll_link"),
            ),
            fields=("force",),
            reduce="netforce",
            track_air_time=False,
            history_length=3,
        ),
        VolumePointsCfg(
            name="leg_volume_points",
            entity_name="robot",
            body_names=".*_ankle_roll_link",
            # Capsule endpoints plus their 0.015 m radius, in each foot frame.
            points_generator=Grid3dPointsGeneratorCfg(
                x_min=-0.071,
                x_max=0.147,
                x_num=10,
                y_min=-0.06,
                y_max=0.06,
                y_num=5,
                z_min=-0.0732,
                z_max=-0.0432,
                z_num=2,
            ),
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="left_height_scanner",
            frame=ObjRef(type="body", name="left_ankle_roll_link", entity="robot"),
            pattern=GridPatternCfg(resolution=0.12, size=(0.12, 0.0)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="right_height_scanner",
            frame=ObjRef(type="body", name="right_ankle_roll_link", entity="robot"),
            pattern=GridPatternCfg(resolution=0.12, size=(0.12, 0.0)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        get_agibot_x2_depth_camera_cfg(),
    )
    # Keep the raw mounting and full FOV; G1's crop is specific to its camera.
    camera = next(sensor for sensor in cfg.scene.sensors if sensor.name == "camera")
    camera.noise_pipeline = {
        "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
        "depth_normalization": DepthNormalizationCfg(depth_range=(0.0, 2.5), normalize=True, output_range=(0.0, 1.0)),
    }
    # Eight frames, spaced by five camera samples, plus one sample of delay.
    camera.data_histories = {"distance_to_image_plane_noised": 37}

    motion_reference_sensor_cfg = copy.deepcopy(motion_reference_cfg)
    if dataset_dir is not None:
        dataset_path = Path(dataset_dir).expanduser().resolve()
        buffer_cfg = motion_reference_sensor_cfg.motion_buffers["run_walk"]
        buffer_cfg.path = str(dataset_path)
        buffer_cfg.filtered_motion_selection_filepath = str(dataset_path / "parkour_motion_without_run_clips.yaml")
    existing_sensors = tuple(
        sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name != motion_reference_sensor_cfg.name
    )
    cfg.scene.sensors = existing_sensors + (motion_reference_sensor_cfg,)

    # MDP settings
    cfg.commands = {
        "base_velocity": PoseVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(8.0, 12.0),
            debug_vis=False,
            velocity_control_stiffness=2.0,
            heading_control_stiffness=2.0,
            rel_standing_envs=0.05,
            ranges=PoseVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
            ),
            random_velocity_terrain=["perlin_rough_stand"],
            velocity_ranges={
                "perlin_rough": {"lin_vel_x": (0.45, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
                "square_gaps": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "pyramid_stairs": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "pyramid_stairs_high": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "pyramid_stairs_inv": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "pyramid_stairs_inv_high": {
                    "lin_vel_x": (0.45, 0.8),
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-1.0, 1.0),
                },
                "boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "mesh_boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "hf_pyramid_slope_inv": {
                    "lin_vel_x": (0.45, 0.8),
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-1.0, 1.0),
                },
            },
            only_positive_lin_vel_x=True,
            lin_vel_threshold=0.0,
            ang_vel_threshold=0.0,
            target_dis_threshold=0.4,
        ),
    }
    policy_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=8,
            flatten_history_dim=True,
        ),
        "velocity_commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=8,
            flatten_history_dim=True,
            noise=None,
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=".*",
                ),
            },
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=8,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=".*",
                ),
            },
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        ),
        "actions": ObservationTermCfg(
            func=envs_mdp.last_action,
            history_length=8,
            flatten_history_dim=True,
        ),
        "depth_image": ObservationTermCfg(
            func=envs_mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        ),
    }
    critic_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            history_length=8,
            flatten_history_dim=True,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            history_length=8,
            flatten_history_dim=True,
        ),
        "velocity_commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=8,
            flatten_history_dim=True,
            noise=None,
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=".*",
                ),
            },
            history_length=8,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=".*",
                ),
            },
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        ),
        "actions": ObservationTermCfg(
            func=envs_mdp.last_action,
            history_length=8,
            flatten_history_dim=True,
        ),
        "depth_image": ObservationTermCfg(
            func=envs_mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        ),
    }
    cfg.observations = {}
    cfg.observations["policy"] = ObservationGroupCfg(
        terms=policy_terms,
        concatenate_terms=False,
        enable_corruption=True,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )

    amp_policy_terms = {
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    preserve_order=True,
                )
            },
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    preserve_order=True,
                )
            },
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
            history_length=10,
            flatten_history_dim=True,
        ),
    }
    amp_reference_terms = {
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity_reference_as_state,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel_reference_as_state,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
                "robot_cfg": SceneEntityCfg(name="robot"),
            },
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel_reference_as_state,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
                "robot_cfg": SceneEntityCfg(name="robot"),
            },
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel_reference_as_state,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel_reference_as_state,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
            history_length=10,
            flatten_history_dim=True,
        ),
    }
    cfg.observations["amp_policy"] = ObservationGroupCfg(
        terms=amp_policy_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )
    cfg.observations["amp_reference"] = ObservationGroupCfg(
        terms=amp_reference_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )

    cfg.rewards = MultiRewardCfg({
        "rewards": {
            # ---------- Task rewards ----------
            "track_lin_vel_xy_exp": RewardTermCfg(
                func=parkour_mdp.track_lin_vel_xy_exp,
                weight=2.0,
                params={"command_name": "base_velocity", "std": 0.5},
            ),
            "track_ang_vel_z_exp": RewardTermCfg(
                func=parkour_mdp.track_ang_vel_z_exp,
                weight=2.0,
                params={"command_name": "base_velocity", "std": 0.5},
            ),
            "heading_error": RewardTermCfg(
                func=parkour_mdp.heading_error,
                weight=-1.0,
                params={"command_name": "base_velocity"},
            ),
            "dont_wait": RewardTermCfg(
                func=parkour_mdp.dont_wait,
                weight=-0.5,
                params={"command_name": "base_velocity"},
            ),
            "is_alive": RewardTermCfg(func=envs_mdp.is_alive, weight=3.0),
            "stand_still": RewardTermCfg(
                func=parkour_mdp.stand_still,
                weight=-0.3,
                params={"command_name": "base_velocity", "offset": 4.0},
            ),
            # ---------- Regularization rewards ----------
            "volume_points_penetration": RewardTermCfg(
                func=parkour_mdp.volume_points_penetration,
                weight=-4.0,
                params={"sensor_name": "leg_volume_points"},
            ),
            "feet_air_time": RewardTermCfg(
                func=parkour_mdp.feet_air_time,
                weight=0.5,
                params={
                    "command_name": "base_velocity",
                    "sensor_name": "contact_forces",
                    "vel_threshold": 0.15,
                },
            ),
            "feet_slide": RewardTermCfg(
                func=parkour_mdp.feet_slide,
                weight=-0.4,
                params={
                    "sensor_name": "contact_forces",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                    ),
                    "threshold": 1.0,
                },
            ),
            "joint_deviation_hip": RewardTermCfg(
                func=parkour_mdp.joint_deviation_square,
                weight=-0.5,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=(".*_hip_yaw_joint", ".*_hip_roll_joint"),
                    )
                },
            ),
            "ang_vel_xy_l2": RewardTermCfg(func=parkour_mdp.ang_vel_xy_l2, weight=-0.05),
            "dof_torques_l2": RewardTermCfg(
                func=parkour_mdp.joint_torques_l2,
                weight=-1.5e-7,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"),
                    )
                },
            ),
            "dof_acc_l2": RewardTermCfg(
                func=envs_mdp.joint_acc_l2,
                weight=-1.25e-7,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            ),
            "dof_vel_l2": RewardTermCfg(
                func=envs_mdp.joint_vel_l2,
                weight=-1e-4,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            ),
            "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.005),
            "flat_orientation_l2": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-3.0),
            "pelvis_orientation_l2": RewardTermCfg(
                func=parkour_mdp.link_orientation,
                weight=-3.0,
                params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
            ),
            "feet_flat_ori": RewardTermCfg(
                func=parkour_mdp.feet_orientation_contact,
                weight=-0.4,
                params={
                    "sensor_name": "contact_forces",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                    ),
                },
            ),
            "feet_at_plane": RewardTermCfg(
                func=parkour_mdp.feet_at_plane,
                weight=-0.1,
                params={
                    "contact_sensor_name": "contact_forces",
                    "left_height_scanner_name": "left_height_scanner",
                    "right_height_scanner_name": "right_height_scanner",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                    ),
                    "height_offset": 0.0732,
                },
            ),
            "feet_close_xy": RewardTermCfg(
                func=parkour_mdp.feet_close_xy_gauss,
                weight=0.4,
                params={
                    "threshold": 0.12,
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                    ),
                    "std": math.sqrt(0.05),
                },
            ),
            "energy": RewardTermCfg(
                func=parkour_mdp.motors_power_square,
                weight=-5e-5,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"),
                    ),
                    "normalize_by_stiffness": True,
                },
            ),
            "freeze_upper_body": RewardTermCfg(
                func=parkour_mdp.joint_deviation_l1,
                weight=-0.004,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=(".*_shoulder_.*", ".*_elbow_joint", ".*_wrist.*", "waist_.*"),
                    )
                },
            ),
            # ---------- Safety rewards ----------
            "dof_pos_limits": RewardTermCfg(
                func=envs_mdp.joint_pos_limits,
                weight=-1.0,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            ),
            "dof_vel_limits": RewardTermCfg(
                func=parkour_mdp.joint_vel_limits,
                weight=-1.0,
                params={
                    "soft_ratio": 0.9,
                    "velocity_limits": X2_JOINT_VELOCITY_LIMITS,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                },
            ),
            "torque_limits": RewardTermCfg(
                func=parkour_mdp.applied_torque_limits_by_ratio,
                weight=-0.01,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                    "limit_ratio": 0.8,
                },
            ),
            "undesired_contacts": RewardTermCfg(
                func=parkour_mdp.undesired_contacts,
                weight=-1.0,
                params={"sensor_name": "undesired_contact_forces", "threshold": 1.0},
            ),
        }
    })
    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=parkour_mdp.tracking_exp_vel,
            params={
                "lin_vel_threshold": (0.3, 0.6),
                "ang_vel_threshold": (0.0, 0.0),
            },
        ),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "terrain_out_bound": TerminationTermCfg(
            func=envs_mdp.terrain_out_of_bounds,
            time_out=True,
            params={"distance_buffer": 2.0},
        ),
        "base_contact": TerminationTermCfg(
            func=parkour_mdp.illegal_contact,
            params={"sensor_name": "torso_contact_forces", "threshold": 1.0},
        ),
        "bad_orientation": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": 1.0},
        ),
        "root_height": TerminationTermCfg(
            func=parkour_mdp.root_height_below_env_origin_minimum,
            params={"minimum_height": 0.5},
        ),
        "dataset_exhausted": TerminationTermCfg(
            func=envs_mdp.dataset_exhausted,
            time_out=True,
            params={
                "reference_cfg": SceneEntityCfg(name="motion_reference"),
                "print_reason": False,
                "reset_without_notice": True,
            },
        ),
    }
    cfg.events = {
        "physics_material": EventTermCfg(
            func=parkour_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "static_friction_range": (0.3, 1.6),
                "dynamic_friction_range": (0.3, 1.6),
                "make_consistent": True,
            },
        ),
        # reset
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
                "velocity_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "z": (-0.2, 0.2),
                    "roll": (-0.2, 0.2),
                    "pitch": (-0.2, 0.2),
                    "yaw": (-0.2, 0.2),
                },
            },
        ),
        "register_virtual_obstacles": EventTermCfg(
            func=envs_mdp.register_virtual_obstacle_to_sensor,
            mode="startup",
            params={
                "sensor_cfgs": SceneEntityCfg("leg_volume_points"),
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.15, 0.15),
                "velocity_range": (0.0, 0.0),
            },
        ),
    }

    if play:
        cfg.scene.num_envs = 10
        cfg.scene.env_spacing = 2.5
        cfg.episode_length_s = 10.0

        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        cfg.scene.terrain.terrain_generator.num_rows = 4
        cfg.scene.terrain.terrain_generator.num_cols = 10

        leg_volume_points_sensor = next(
            sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name == "leg_volume_points"
        )
        leg_volume_points_sensor.debug_vis = True

        cfg.scene.terrain.collision_debug_vis = False
        cfg.commands["base_velocity"].debug_vis = True
        cfg.commands["base_velocity"].patch_vis = False
        cfg.terminations["root_height"] = None
        cfg.events["physics_material"] = None
        cfg.events["reset_robot_joints"].params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
    return cfg


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def instinct_x2_parkour_amp_final_cfg(
    *,
    play: bool = False,
    dataset_dir: str | Path | None = None,
) -> X2ParkourAmpEnvCfg:
    """Create the final X2 parkour AMP env configuration.

    Args:
      play: If True, apply play-mode overrides (fewer envs, relaxed
        termination, etc.).
      dataset_dir: Directory containing retargeted clips and their selection YAML.

    Returns:
      A fully-built ``X2ParkourAmpEnvCfg`` instance.
    """
    # Build base parkour config (already includes play overrides if requested)
    cfg = instinct_x2_parkour_amp_env_cfg(play=play, dataset_dir=dataset_dir)

    # Apply play-mode viewer overrides
    if play:
        cfg.viewer = ViewerConfig(
            lookat=(0.0, 0.75, 0.0),
            distance=4.123105625617661,
            elevation=-14.036243467926479,
            azimuth=180.0,
            origin_type=ViewerConfig.OriginType.WORLD,
            entity_name=None,
        )
        cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
        cfg.viewer.entity_name = None
        cfg.viewer.body_name = None

    return cfg
