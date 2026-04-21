#!/usr/bin/env python3
import os
import math
import time
import numpy as np
import scipy.spatial.transform 

from oculus_reader import OculusReader

#ros2 department
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from transformations import quaternion_from_euler, quaternion_from_matrix, euler_from_quaternion
from rclpy.executors import MultiThreadedExecutor

from scipy.spatial.transform import Rotation
    
def matrix2xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    pitch = np.arcsin(-matrix[2, 0])
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]

def xyzrpy_to_mat(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    mat[:3, 3] = np.array([x, y, z])
    return mat

def xyzquat2Mat(pos, quat):
    # 如果输入是列表，直接赋值
    # pos 预期为 [x, y, z]
    # quat 预期为 [x, y, z, w]
    p = pos 
    q = quat
    
    rotation_matrix = scipy.spatial.transform.Rotation.from_quat(q).as_matrix()
    
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = p
    
    return matrix

def mat2xyzquat(matrix):
    # 提取位置
    pos = matrix[:3, 3]
    
    # 提取旋转矩阵并转换为四元数
    rotation_matrix = matrix[:3, :3]
    quat = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix).as_quat()
    
    return pos, quat

def calc_pose_incre(start_pose_matrix, current_pose_matrix, zero_matrix):
    end_matrix = xyzrpy_to_mat(current_pose_matrix[0], current_pose_matrix[1], current_pose_matrix[2],
                               current_pose_matrix[3], current_pose_matrix[4], current_pose_matrix[5])
    result_matrix = np.dot(zero_matrix, np.dot(np.linalg.inv(start_pose_matrix), end_matrix))
    
    pos, quat = mat2xyzquat(result_matrix)
    
    return pos, quat


class RosOperator(Node):
    def __init__(self):
        super().__init__('teleop_single_nero_node')
        
        #声明并获取参数
        self.pub_delta_pose = self.create_publisher(PoseStamped, '/delta_pose', 10)
        self.pub_move_j = self.create_publisher(JointState, '/control/joint_states', 10)

        # 回调前先初始化，避免启动阶段访问到不存在的属性
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
        
        
        # 订阅回调        
        self.create_subscription(
            PoseStamped, 
            '/right_handle_pose', 
            self.handle_pose_callback, 
            1)
        self.create_subscription(
            PoseStamped, 
            '/feedback/tcp_pose', 
            self.tcp_pose_callback, 
            1)
        
        # 这里可选为 WIFI连接 或 USB连接
        # self.oculus_reader = OculusReader(ip_address='192.168.124.2')    #  WIFI连接
        self.oculus_reader = OculusReader()                         #  USB连接
        
        # 延时0.5秒，确保 OculusReader 初始化完成   
        time.sleep(0.5)
        
        # 默认单位位姿，收到 /feedback/tcp_pose 后再更新到真实零点
        self.zero_matrix = xyzrpy_to_mat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.start_pose_matrix = self.zero_matrix

        # 控制频率 30Hz
        self.control_timer = self.create_timer(1.0/30.0 , self.control_loop)  # 30Hz 控制频率

    # 手柄位姿回调函数，更新当前手柄位姿
    def handle_pose_callback(self, msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.z = msg.pose.position.z
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    
    # 机械臂位姿回调函数，更新当前机械臂位姿
    def tcp_pose_callback(self, msg):
        self.tcp_x = msg.pose.position.x
        self.tcp_y = msg.pose.position.y
        self.tcp_z = msg.pose.position.z
        (self.tcp_roll, self.tcp_pitch, self.tcp_yaw) = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

        # 启动时首次收到反馈后，用真实 TCP 位姿更新零点/起点
        if self.start_pose_matrix is self.zero_matrix:
            self.zero_matrix = xyzrpy_to_mat(self.tcp_x, self.tcp_y, self.tcp_z, self.tcp_roll, self.tcp_pitch, self.tcp_yaw)
            self.start_pose_matrix = self.zero_matrix
        
    def control_loop(self):
        _, buttons = self.oculus_reader.get_transformations_and_buttons()
        
        # --------------- piper夹爪控制部分 ---------------
        right_trig = float(buttons.get('rightTrig', [0.0])[0]) if buttons.get('rightTrig') else 0.0
        gripper_value = right_trig * 0.07  # right_trig 最大值为1.0，映射到0.07，因为夹爪最大行程为0.07m
        gripper_msg = JointState()
        gripper_msg.header = Header()
        gripper_msg.header.stamp = self.get_clock().now().to_msg()
        gripper_msg.name = ['gripper']
        gripper_msg.position = [gripper_value]
        self.pub_move_j.publish(gripper_msg)
        # --------------- piper夹爪控制部分 ---------------
        
        # 还未收到手柄位姿，直接跳过本周期
        if self.x is None:
            return
                        
        current_pose = [self.x,self.y,self.z,self.roll,self.pitch,self.yaw]
            
        if buttons.get('A', False):
            self.start_pose_matrix = xyzrpy_to_mat(self.x,self.y,self.z,self.roll,self.pitch,self.yaw)
            self.flag = True
            print("开始遥操作")
            
        
        if buttons.get('B', False):
            self.flag = False
            print("停止遥操作")
            if self.tcp_x is not None:
                self.zero_matrix = xyzrpy_to_mat(self.tcp_x, self.tcp_y, self.tcp_z, self.tcp_roll, self.tcp_pitch, self.tcp_yaw)
                # print("更新零点位姿")
                self.start_pose_matrix = self.zero_matrix
                # print("更新起始位姿")
        
        # 如果self.flag为True，计算增量位姿并将其转换为posestamped消息发布
        if self.flag == True :
            xyz, quat = calc_pose_incre(self.start_pose_matrix, current_pose, self.zero_matrix)
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'vr_device'
            pose_msg.pose.position.x = xyz[0]
            pose_msg.pose.position.y = xyz[1]
            pose_msg.pose.position.z = xyz[2]
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            self.pub_delta_pose.publish(pose_msg)
        
def main(args=None):
    rclpy.init(args=args)
    try:
        teleop_node = RosOperator()
        # 使用多线程执行器
        executor = MultiThreadedExecutor()
        executor.add_node(teleop_node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
