#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime


class ImageSaver(Node):

    def __init__(self):
        super().__init__('image_saver')

        self.save_dir = "/home/ubuntu/Deepak_WS/Roboreg_test/franka_top"
        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.image_count = 0
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

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"frame_{timestamp}.png"
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
