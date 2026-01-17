#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import os
from datetime import datetime


class DepthImageSaver(Node):

    def __init__(self):
        super().__init__('depth_image_saver')

        self.save_dir = "/home/ubuntu/Deepak_WS/Roboreg_test/franka_top"
        os.makedirs(self.save_dir, exist_ok=True)

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',   #Change camera topic for TOP CAM
            self.depth_callback,
            10
        )

        self.saved = False
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

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filepath = os.path.join(
                self.save_dir, f"depth_m_{timestamp}.npy"
            )

            np.save(filepath, depth_m)

            self.get_logger().info(
                f"Saved depth image (meters) to: {filepath} "
                f"Shape: {depth_m.shape}, dtype: {depth_m.dtype}"
            )

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
