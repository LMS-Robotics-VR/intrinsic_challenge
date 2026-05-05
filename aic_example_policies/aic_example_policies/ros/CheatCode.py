#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

class CheatCode(Policy):
    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        super().__init__(parent_node)

    def _ease_in_out(self, t: float) -> float:
        """Smoothstep polynomial (3t^2 - 2t^3) to prevent jerk penalties."""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())
                return True
            except TransformException:
                self.sleep_for(0.1)
        return False

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
        freeze_integrator: bool = False,
    ) -> Pose:
        q_port = (port_transform.rotation.w, port_transform.rotation.x,
                  port_transform.rotation.y, port_transform.rotation.z)
        
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link", f"{self._task.cable_name}/{self._task.plug_name}_link", Time())
        q_plug = (plug_tf_stamped.transform.rotation.w, plug_tf_stamped.transform.rotation.x,
                  plug_tf_stamped.transform.rotation.y, plug_tf_stamped.transform.rotation.z)
        
        q_diff = quaternion_multiply(q_port, (-q_plug[0], q_plug[1], q_plug[2], q_plug[3]))
        
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
        q_gripper = (gripper_tf_stamped.transform.rotation.w, gripper_tf_stamped.transform.rotation.x,
                     gripper_tf_stamped.transform.rotation.y, gripper_tf_stamped.transform.rotation.z)
        
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (gripper_tf_stamped.transform.translation.x,
                       gripper_tf_stamped.transform.translation.y,
                       gripper_tf_stamped.transform.translation.z)
        port_xy = (port_transform.translation.x, port_transform.translation.y)
        plug_xyz = (plug_tf_stamped.transform.translation.x,
                    plug_tf_stamped.transform.translation.y,
                    plug_tf_stamped.transform.translation.z)
        
        tip_gripper_z_offset = gripper_xyz[2] - plug_xyz[2]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        elif not freeze_integrator:
            self._tip_x_error_integrator = np.clip(self._tip_x_error_integrator + (port_xy[0] - plug_xyz[0]), 
                                                   -self._max_integrator_windup, self._max_integrator_windup)
            self._tip_y_error_integrator = np.clip(self._tip_y_error_integrator + (port_xy[1] - plug_xyz[1]), 
                                                   -self._max_integrator_windup, self._max_integrator_windup)

        i_gain = 0.15
        target_x = port_xy[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + i_gain * self._tip_y_error_integrator
        target_z = port_transform.translation.z + z_offset - tip_gripper_z_offset

        return Pose(
            position=Point(x=position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
                           y=position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
                           z=position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2]),
            orientation=Quaternion(w=q_gripper_slerp[0], x=q_gripper_slerp[1], y=q_gripper_slerp[2], z=q_gripper_slerp[3])
        )

    def insert_cable(self, task: Task, get_observation: GetObservationCallback, move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback):
        self.get_logger().info(f"Executing Fast Unified Strategy. Target: {task.port_type}")
        self._task = task
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        if not self._wait_for_tf("base_link", port_frame) or not self._wait_for_tf("base_link", cable_tip_frame):
            return False

        port_transform = self._parent_node._tf_buffer.lookup_transform("base_link", port_frame, Time()).transform

        # Check port_type to know physically when to stop pushing.
        if task.port_type == "sc":
            final_insertion_depth = -0.008
        else:
            final_insertion_depth = -0.012

        clearance_z = 0.25

        # --- Fast Traverse ---
        steps_traverse = 60 # Reduced steps to speed up the fly-over
        for t in range(1, steps_traverse + 1):
            frac = self._ease_in_out(t / float(steps_traverse))
            try:
                self.set_pose_target(move_robot=move_robot, pose=self.calc_gripper_pose(
                    port_transform, slerp_fraction=frac, position_fraction=frac, 
                    z_offset=clearance_z, reset_xy_integrator=True))
            except TransformException: pass
            self.sleep_for(0.04) # Faster control loop rate

        # The Two-Stage Drop
        z_offset = clearance_z
        
        while z_offset > final_insertion_depth:
            # Stage A: Open Air - Fast Drop
            if z_offset > 0.03:
                z_offset -= 0.005 
            # Stage B: The Socket - Precision Push
            else:
                z_offset -= 0.0005 # Gently push the connector in
                
            try:
                self.set_pose_target(move_robot=move_robot, pose=self.calc_gripper_pose(
                    port_transform, z_offset=z_offset, reset_xy_integrator=False))
            except TransformException: pass
            self.sleep_for(0.04)

        self.get_logger().info("Stabilizing...")
        self.sleep_for(5.0) 
        
        self.get_logger().info("Task complete.")
        return True