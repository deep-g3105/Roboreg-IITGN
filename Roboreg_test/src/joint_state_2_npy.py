#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import os
from datetime import datetime
import re


class JointPositionSaver(Node):
    def _get_start_index(self) -> int:
        pattern = re.compile(r"franka_joints_(\d+)\.npy")
        max_index = -1

        for fname in os.listdir(self.save_dir):
            match = pattern.match(fname)
            if match:
                idx = int(match.group(1))
                max_index = max(max_index, idx)

        return max_index + 1


    def __init__(self):
        super().__init__('joint_position_saver')

        self.save_dir = "/home/ubuntu/Deepak_WS/Roboreg-IITGN/Roboreg_test/franka_side"
        os.makedirs(self.save_dir, exist_ok=True)

        # ===== DEFINE CORRECT JOINT ORDER HERE =====
        self.joint_order = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7"
        ]
        # ==========================================

        self.subscription = self.create_subscription(
            JointState,
            '/NS_1/franka/joint_states', # for heal use '/joint_states' , for franka use '/NS_1/franka/joint_states'
            self.joint_state_callback,
            10
        )
        self.file_index = self._get_start_index()
        self.saved = False
        self.get_logger().info("Waiting for /joint_states...")
        self.get_logger().info(f"Starting joint file index at {self.file_index}")

    def joint_state_callback(self, msg: JointState):
        if self.saved:
            return

        # Build name -> position map
        joint_map = dict(zip(msg.name, msg.position))

        ordered_positions = []

        for joint in self.joint_order:
            if joint not in joint_map:
                self.get_logger().error(
                    f"Joint '{joint}' not found in /joint_states"
                )
                return
            ordered_positions.append(joint_map[joint])

        ordered_positions = np.array(ordered_positions, dtype=np.float64)
        filename = f"franka_joints_{self.file_index}.npy"
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(
            self.save_dir, filename
        )

        np.save(filepath, ordered_positions)
        self.get_logger().info(f"Saved ordered joint positions to: {filepath}")
        
        self.file_index += 1
        self.saved = True
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = JointPositionSaver()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    main()
