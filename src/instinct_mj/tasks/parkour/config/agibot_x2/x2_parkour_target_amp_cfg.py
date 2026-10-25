"""Agibot X2 parkour sensor configuration.

Camera settings belong to the task, following the G1 parkour configuration.
A complete X2 environment and task registration are not defined yet.
"""

from __future__ import annotations

import math

from mjlab.sensor import ObjRef, PinholeCameraPatternCfg

from instinct_mj.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg

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
