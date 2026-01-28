#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import os
from datetime import datetime
import re

class DepthImageSaver(Node):

    def _get_start_index(self) -> int:
        pattern = re.compile(r"heal_depth_(\d+)\.npy") #for heal use 'heal_depth_(\d+)\.npy' , for franka use 'franka_depth_(\d+)\.npy'
        max_index = -1

        for fname in os.listdir(self.save_dir):
            match = pattern.match(fname)
            if match:
                idx = int(match.group(1))
                max_index = max(max_index, idx)

        return max_index + 1

    def __init__(self):
        super().__init__('depth_image_saver')

        self.save_dir = "/home/ubuntu/Deepak_WS/Roboreg-IITGN/Roboreg_test/heal_top"
        os.makedirs(self.save_dir, exist_ok=True)

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',   #Change camera topic for TOP CAM
            self.depth_callback,
            10
        )

        self.image_index = self._get_start_index()
        self.saved = False
        self.get_logger().info(f"Starting depth index at {self.image_index}")
        self.get_logger().info("Waiting for depth image...")

    def depth_callback(self, msg: Image):
        if self.saved:
            return

        try:
            # Raw depth in millimeters (uint16)
            depth_mm = np.frombuffer(
                msg.data,
                dtype=np.uint16
            ).reshape(msg.height, msg.width)

            # Convert to meters (float32)
            depth_m = depth_mm.astype(np.float32) * 0.001
            filename = f"heal_depth_{self.image_index}.npy"
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filepath = os.path.join(
                self.save_dir, filename
            )

            np.save(filepath, depth_m)

            self.get_logger().info(
                f"Saved depth image (meters) to: {filepath} "
                f"Shape: {depth_m.shape}, dtype: {depth_m.dtype}"
            )
            self.image_index += 1
            self.saved = True
            rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Failed to save depth image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DepthImageSaver()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    main()
