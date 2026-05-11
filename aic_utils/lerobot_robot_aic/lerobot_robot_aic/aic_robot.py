#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#

from lerobot.cameras import CameraConfig
from lerobot_robot_ros import ROS2CameraConfig

arm_joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Capturing only Left (as Global) and Right (as Wrist) for ForceVLA
aic_cameras: dict[str, CameraConfig] = {
    "left_camera": ROS2CameraConfig(
        name="left_camera",
        fps=20,
        width=1152,
        height=1024,
        topic="/left_camera/image",
    ),
    "right_camera": ROS2CameraConfig(
        name="right_camera",
        fps=20,
        width=1152,
        height=1024,
        topic="/right_camera/image",
    ),
}