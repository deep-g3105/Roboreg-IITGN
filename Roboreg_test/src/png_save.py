#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime
import re


class ImageSaver(Node):

    def _get_start_index(self) -> int:
        pattern = re.compile(r"franka_image_(\d+)\.png")
        max_index = -1

        for fname in os.listdir(self.save_dir):
            match = pattern.match(fname)
            if match:
                idx = int(match.group(1))
                max_index = max(max_index, idx)

        return max_index + 1

    def __init__(self):
        super().__init__('image_saver')

        self.save_dir = "/home/ubuntu/Deepak_WS/Roboreg-IITGN/Roboreg_test/franka_top"
        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.image_count = self._get_start_index()
        self.get_logger().info(f"Starting image index at {self.image_count}")
        self.max_images = 1

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  #Change camera topic for TOP CAM
            self.image_callback,
            10
        )

        self.get_logger().info(
            f"Saving {self.max_images} images to: {self.save_dir}"
        )

    def image_callback(self, msg: Image):
        if self.image_count >= self.max_images:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )

            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"franka_image_{self.image_count}.png"
            filepath = os.path.join(self.save_dir, filename)

            cv2.imwrite(filepath, cv_image)
            self.image_count += 1

            self.get_logger().info(
                f"Saved {self.image_count}/{self.max_images}: {filepath}"
            )

            # Shutdown after saving required images
            if self.image_count >= self.max_images:
                self.get_logger().info("Saved required images. Shutting down...")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
