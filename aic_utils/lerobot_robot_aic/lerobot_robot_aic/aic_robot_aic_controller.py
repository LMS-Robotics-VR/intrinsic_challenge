import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from threading import Thread
from typing import Any, Callable, TypedDict, cast

import cv2
import numpy as np
import rclpy
from aic_control_interfaces.msg import (
    ControllerState,
    JointMotionUpdate,
    MotionUpdate,
    TargetMode,
    TrajectoryGenerationMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Twist, Vector3, Wrench, WrenchStamped
from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from numpy.typing import NDArray
from rclpy.client import Client
from rclpy.executors import SingleThreadedExecutor
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_sensor_data
from rclpy.subscription import Subscription
from sensor_msgs.msg import JointState

from .aic_robot import aic_cameras, arm_joint_names
from .types import JointMotionUpdateActionDict, MotionUpdateActionDict

logger = logging.getLogger(__name__)

ObservationState = TypedDict(
    "ObservationState",
    {
        "tcp_pose.position.x": float, "tcp_pose.position.y": float, "tcp_pose.position.z": float,
        "tcp_pose.orientation.x": float, "tcp_pose.orientation.y": float, "tcp_pose.orientation.z": float, "tcp_pose.orientation.w": float,
        "tcp_velocity.linear.x": float, "tcp_velocity.linear.y": float, "tcp_velocity.linear.z": float,
        "tcp_velocity.angular.x": float, "tcp_velocity.angular.y": float, "tcp_velocity.angular.z": float,
        "tcp_error.x": float, "tcp_error.y": float, "tcp_error.z": float,
        "tcp_error.rx": float, "tcp_error.ry": float, "tcp_error.rz": float,
        "joint_positions.0": float, "joint_positions.1": float, "joint_positions.2": float,
        "joint_positions.3": float, "joint_positions.4": float, "joint_positions.5": float, "joint_positions.6": float,
        "observation.force.fx": float, "observation.force.fy": float, "observation.force.fz": float,
        "observation.force.tx": float, "observation.force.ty": float, "observation.force.tz": float,
    },
)

class CameraImageScaling(TypedDict):
    left_camera: float
    right_camera: float

@RobotConfig.register_subclass("aic_controller")
@dataclass(kw_only=True)
class AICRobotAICControllerConfig(RobotConfig):
    teleop_target_mode: str = "cartesian"  
    teleop_frame_id: str = "gripper/tcp"  
    arm_joint_names: list[str] = field(default_factory=arm_joint_names.copy)
    cameras: dict[str, CameraConfig] = field(default_factory=aic_cameras.copy)
    camera_image_scaling: CameraImageScaling = field(
        default_factory=lambda: {"left_camera": 0.25, "right_camera": 0.25}
    )

@dataclass(kw_only=True)
class AICRos2Interface:
    node: Node
    executor: SingleThreadedExecutor
    executor_thread: Thread
    change_target_mode_client: Client[ChangeTargetMode.Request, ChangeTargetMode.Response]
    motion_update_pub: Publisher[MotionUpdate]
    joint_motion_update_pub: Publisher[JointMotionUpdate]
    controller_state_sub: Subscription[ControllerState]
    joint_states_sub: Subscription[JointState]
    wrench_sub: Subscription[WrenchStamped]
    action_sub: Subscription[MotionUpdate]
    logger: RcutilsLogger

    @staticmethod
    def connect(
        controller_state_cb: Callable[[ControllerState], None],
        joint_states_cb: Callable[[JointState], None],
        wrench_cb: Callable[[WrenchStamped], None],
        action_cb: Callable[[MotionUpdate], None],
    ) -> "AICRos2Interface":
        if not rclpy.ok(): rclpy.init()

        node = Node("aic_robot_node")
        logger = node.get_logger()
        logger.set_level(logging.DEBUG)

        change_target_mode_client = node.create_client(ChangeTargetMode, f"/aic_controller/change_target_mode")
        while not change_target_mode_client.wait_for_service():
            time.sleep(1.0)

        motion_update_pub = node.create_publisher(MotionUpdate, "/aic_controller/pose_commands", 10)
        joint_motion_update_pub = node.create_publisher(JointMotionUpdate, "/aic_controller/joint_commands", 10)

        controller_state_sub = node.create_subscription(ControllerState, "/aic_controller/controller_state", controller_state_cb, 10)
        joint_states_sub = node.create_subscription(JointState, "/joint_states", joint_states_cb, qos_profile_sensor_data)
        
        wrench_sub = node.create_subscription(WrenchStamped, "/fts_broadcaster/wrench", wrench_cb, qos_profile_sensor_data)
        action_sub = node.create_subscription(MotionUpdate, "/aic_controller/pose_commands", action_cb, 10)

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor_thread = Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        time.sleep(3)  

        return AICRos2Interface(
            node=node, executor=executor, executor_thread=executor_thread,
            change_target_mode_client=change_target_mode_client,
            motion_update_pub=motion_update_pub, joint_motion_update_pub=joint_motion_update_pub,
            controller_state_sub=controller_state_sub, joint_states_sub=joint_states_sub,
            wrench_sub=wrench_sub, action_sub=action_sub, logger=logger,
        )


class AICRobotAICController(Robot):
    name = "ur5e_aic"

    def __init__(self, config: AICRobotAICControllerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.ros2_interface: AICRos2Interface | None = None
        
        self.last_controller_state: ControllerState | None = None
        self.last_joint_states: JointState | None = None
        self.last_wrench_state: WrenchStamped | None = None
        self.last_action_msg: MotionUpdate | None = None

        self._is_connected = False
        self.frame_id = config.teleop_frame_id
        self.teleop_target_mode = config.teleop_target_mode

    @property
    def action_features(self) -> dict:
        return {}

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def send_action(self, action):
        pass

    def send_change_control_mode_req(self, mode: int):
        if not self.ros2_interface: raise DeviceNotConnectedError()
        req = ChangeTargetMode.Request()
        req.target_mode.mode = mode
        self.ros2_interface.change_target_mode_client.call(req)
        time.sleep(0.5)

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (int(self.config.cameras[cam].height * self.config.camera_image_scaling[cam]), int(self.config.cameras[cam].width * self.config.camera_image_scaling[cam]), 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict:
        return {**ObservationState.__annotations__, **self._cameras_ft}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected: raise DeviceAlreadyConnectedError(f"{self} already connected")

        def controller_state_cb(msg: ControllerState): self.last_controller_state = msg
        def joint_states_cb(msg: JointState): self.last_joint_states = msg
        def wrench_cb(msg: WrenchStamped): self.last_wrench_state = msg
        def action_cb(msg: MotionUpdate): self.last_action_msg = msg

        self.ros2_interface = AICRos2Interface.connect(controller_state_cb, joint_states_cb, wrench_cb, action_cb)
        
        mode = TargetMode.MODE_JOINT if self.teleop_target_mode == "joint" else TargetMode.MODE_CARTESIAN
        self.send_change_control_mode_req(mode)

        for cam in self.cameras.values(): cam.connect()
        self._is_connected = True

    def get_latest_action(self) -> list[float] | None:
        if not self.last_action_msg: return None
        if hasattr(self.last_action_msg, 'pose'):
            p = self.last_action_msg.pose
            return [p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        elif hasattr(self.last_action_msg, 'velocity'):
            v = self.last_action_msg.velocity
            return [v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z, 0.0]
        return None

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected: raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.last_controller_state or not self.last_joint_states or not self.last_wrench_state:
            return {}

        tcp_pose = self.last_controller_state.tcp_pose
        tcp_velocity = self.last_controller_state.tcp_velocity
        tcp_error = self.last_controller_state.tcp_error
        joint_positions = self.last_joint_states.position
        fw = self.last_wrench_state.wrench 
        
        controller_state_obs: ObservationState = {
            "tcp_pose.position.x": tcp_pose.position.x, "tcp_pose.position.y": tcp_pose.position.y, "tcp_pose.position.z": tcp_pose.position.z,
            "tcp_pose.orientation.x": tcp_pose.orientation.x, "tcp_pose.orientation.y": tcp_pose.orientation.y, "tcp_pose.orientation.z": tcp_pose.orientation.z, "tcp_pose.orientation.w": tcp_pose.orientation.w,
            "tcp_velocity.linear.x": tcp_velocity.linear.x, "tcp_velocity.linear.y": tcp_velocity.linear.y, "tcp_velocity.linear.z": tcp_velocity.linear.z,
            "tcp_velocity.angular.x": tcp_velocity.angular.x, "tcp_velocity.angular.y": tcp_velocity.angular.y, "tcp_velocity.angular.z": tcp_velocity.angular.z,
            "tcp_error.x": tcp_error[0], "tcp_error.y": tcp_error[1], "tcp_error.z": tcp_error[2], "tcp_error.rx": tcp_error[3], "tcp_error.ry": tcp_error[4], "tcp_error.rz": tcp_error[5],
            "joint_positions.0": joint_positions[0], "joint_positions.1": joint_positions[1], "joint_positions.2": joint_positions[2],
            "joint_positions.3": joint_positions[3], "joint_positions.4": joint_positions[4], "joint_positions.5": joint_positions[5], "joint_positions.6": joint_positions[6],
            "observation.force.fx": fw.force.x, "observation.force.fy": fw.force.y, "observation.force.fz": fw.force.z,
            "observation.force.tx": fw.torque.x, "observation.force.ty": fw.torque.y, "observation.force.tz": fw.torque.z,
        }

        cam_obs: dict[str, NDArray[Any]] = {}
        for cam_key, cam in self.cameras.items():
            try:
                data = cam.async_read(timeout_ms=2000)
                if data is not None and data.size > 0:
                    image_scale = self.config.camera_image_scaling[cam_key]
                    cam_obs[cam_key] = cv2.resize(data, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA) if image_scale != 1 else data
                else:
                    cam_obs[cam_key] = np.zeros(self._cameras_ft[cam_key], dtype=np.uint8)
            except Exception as e:
                logger.error(f"Failed to read camera {cam_key}: {e}")

        return {**cam_obs, **controller_state_obs}

    def disconnect(self) -> None:
        if not self.is_connected: return
        for cam in self.cameras.values(): cam.disconnect()
        if self.ros2_interface:
            self.ros2_interface.node.destroy_node()
            self.ros2_interface.executor.shutdown()
            self.ros2_interface.executor_thread.join()
            self.ros2_interface = None
        self._is_connected = False