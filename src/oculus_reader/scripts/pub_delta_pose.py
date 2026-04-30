#!/usr/bin/env python3
import time
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from oculus_reader import OculusReader
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from transformations import euler_from_quaternion

def xyzrpy_to_mat(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    mat[:3, 3] = np.array([x, y, z])
    return mat

def mat2xyzquat(matrix: np.ndarray):
    pos = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    quat = Rotation.from_matrix(rotation_matrix).as_quat()
    return pos, quat

def calc_pose_incre(start_pose_matrix: np.ndarray, current_pose_matrix, zero_matrix: np.ndarray):
    end_matrix = xyzrpy_to_mat(
        current_pose_matrix[0],
        current_pose_matrix[1],
        current_pose_matrix[2],
        current_pose_matrix[3],
        current_pose_matrix[4],
        current_pose_matrix[5],
    )
    result_matrix = np.dot(zero_matrix, np.dot(np.linalg.inv(start_pose_matrix), end_matrix))
    return mat2xyzquat(result_matrix)

class RosOperator(Node):
    def __init__(self):
        super().__init__("pub_delta_pose_node")

        # Topic parameters
        self.declare_parameter("handle_pose_topic", "/right_handle_pose")
        self.declare_parameter("feedback_tcp_pose_topic", "/feedback/tcp_pose")
        self.declare_parameter("delta_pose_topic", "/delta_pose")
        self.declare_parameter("control_joint_topic", "/control/joint_states")

        # Button mapping parameters
        dynamic_string_param = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("start_button", "A", dynamic_string_param)
        self.declare_parameter("stop_button", "B", dynamic_string_param)
        self.declare_parameter("trigger_axis", "rightTrig")

        # Gripper/control parameters
        self.declare_parameter("gripper_joint_name", "gripper")
        self.declare_parameter("gripper_max_range", 0.07)
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("hand_name", "right")

        handle_pose_topic = str(self.get_parameter("handle_pose_topic").value)
        feedback_tcp_pose_topic = str(self.get_parameter("feedback_tcp_pose_topic").value)
        delta_pose_topic = str(self.get_parameter("delta_pose_topic").value)
        control_joint_topic = str(self.get_parameter("control_joint_topic").value)

        self.start_button = self._normalize_button_name(self.get_parameter("start_button").value, "start_button")
        self.stop_button = self._normalize_button_name(self.get_parameter("stop_button").value, "stop_button")
        self.trigger_axis = str(self.get_parameter("trigger_axis").value)

        self.gripper_joint_name = str(self.get_parameter("gripper_joint_name").value)
        self.gripper_max_range = float(self.get_parameter("gripper_max_range").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.hand_name = str(self.get_parameter("hand_name").value)

        self.pub_delta_pose = self.create_publisher(PoseStamped, delta_pose_topic, 10)
        self.pub_move_j = self.create_publisher(JointState, control_joint_topic, 10)

        # Callback inputs
        self.x = None
        self.y = None
        self.z = None
        self.roll = None
        self.pitch = None
        self.yaw = None

        self.tcp_x = None
        self.tcp_y = None
        self.tcp_z = None
        self.tcp_roll = None
        self.tcp_pitch = None
        self.tcp_yaw = None

        self.flag = False

        self.create_subscription(PoseStamped, handle_pose_topic, self.handle_pose_callback, 1)
        self.create_subscription(PoseStamped, feedback_tcp_pose_topic, self.tcp_pose_callback, 1)

        # Wifi example:
        # self.oculus_reader = OculusReader(ip_address='192.168.124.2')
        self.oculus_reader = OculusReader()

        # Ensure OculusReader has been initialized.
        time.sleep(0.5)

        self.zero_matrix = xyzrpy_to_mat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.start_pose_matrix = self.zero_matrix

        self.control_timer = self.create_timer(1.0 / max(control_rate_hz, 1.0), self.control_loop)

        self.get_logger().info(
            f"pub_delta_pose ready ({self.hand_name}). "
            f"handle_topic={handle_pose_topic}, feedback_topic={feedback_tcp_pose_topic}, "
            f"delta_topic={delta_pose_topic}, control_topic={control_joint_topic}, "
            f"buttons=({self.start_button}/{self.stop_button}), trigger={self.trigger_axis}"
        )

    def handle_pose_callback(self, msg: PoseStamped):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.z = msg.pose.position.z
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        )

    def tcp_pose_callback(self, msg: PoseStamped):
        self.tcp_x = msg.pose.position.x
        self.tcp_y = msg.pose.position.y
        self.tcp_z = msg.pose.position.z
        (self.tcp_roll, self.tcp_pitch, self.tcp_yaw) = euler_from_quaternion(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        )

        # Use first valid TCP feedback as zero pose
        if self.start_pose_matrix is self.zero_matrix:
            self.zero_matrix = xyzrpy_to_mat(
                self.tcp_x,
                self.tcp_y,
                self.tcp_z,
                self.tcp_roll,
                self.tcp_pitch,
                self.tcp_yaw,
            )
            self.start_pose_matrix = self.zero_matrix

    def _extract_trigger_value(self, buttons: dict) -> float:
        trigger_raw: Any = buttons.get(self.trigger_axis, [0.0])
        if isinstance(trigger_raw, (list, tuple)):
            return float(trigger_raw[0]) if trigger_raw else 0.0
        if isinstance(trigger_raw, (int, float)):
            return float(trigger_raw)
        return 0.0

    def _normalize_button_name(self, raw_value: Any, param_name: str) -> str:
        if isinstance(raw_value, bool):
            normalized_bool = "Y" if raw_value else "N"
            self.get_logger().warn(
                f"Parameter '{param_name}' is BOOL({raw_value}), normalized to '{normalized_bool}'. "
                "Please pass string button names in launch/params."
            )
            return normalized_bool

        normalized = str(raw_value).strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in ("'", '"'):
            normalized = normalized[1:-1].strip()

        normalized = normalized.upper()
        allowed = {"A", "B", "X", "Y", "N"}
        if normalized not in allowed:
            raise ValueError(
                f"Invalid {param_name}='{raw_value}'. Expected one of {sorted(allowed)} or BOOL for compatibility."
            )
        return normalized

    def control_loop(self):
        _, buttons = self.oculus_reader.get_transformations_and_buttons()

        # Gripper control
        trigger_value = self._extract_trigger_value(buttons)
        gripper_value = max(0.0, min(trigger_value, 1.0)) * self.gripper_max_range
        gripper_msg = JointState()
        gripper_msg.header = Header()
        gripper_msg.header.stamp = self.get_clock().now().to_msg()
        gripper_msg.name = [self.gripper_joint_name]
        gripper_msg.position = [gripper_value]
        self.pub_move_j.publish(gripper_msg)

        if self.x is None:
            return

        current_pose = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

        if buttons.get(self.start_button, False):
            if not self.flag:
                self.get_logger().info(f"[{self.hand_name}] 开始遥操作")
            self.start_pose_matrix = xyzrpy_to_mat(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)
            self.flag = True

        if buttons.get(self.stop_button, False):
            if self.flag:
                self.get_logger().info(f"[{self.hand_name}] 停止遥操作")
            self.flag = False
            if self.tcp_x is not None:
                self.zero_matrix = xyzrpy_to_mat(
                    self.tcp_x,
                    self.tcp_y,
                    self.tcp_z,
                    self.tcp_roll,
                    self.tcp_pitch,
                    self.tcp_yaw,
                )
                self.start_pose_matrix = self.zero_matrix

        if self.flag:
            xyz, quat = calc_pose_incre(self.start_pose_matrix, current_pose, self.zero_matrix)
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "vr_device"
            pose_msg.pose.position.x = float(xyz[0])
            pose_msg.pose.position.y = float(xyz[1])
            pose_msg.pose.position.z = float(xyz[2])
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])
            self.pub_delta_pose.publish(pose_msg)

    def destroy_node(self):
        self.oculus_reader.stop()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    teleop_node = None
    try:
        teleop_node = RosOperator()
        executor = MultiThreadedExecutor()
        executor.add_node(teleop_node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if teleop_node is not None:
            teleop_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
