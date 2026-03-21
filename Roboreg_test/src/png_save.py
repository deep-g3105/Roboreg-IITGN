#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import re
import queue
import threading


class ImageSaver(Node):

    def _get_start_index(self) -> int:
        """
        Scan save directory and find the next available franka_image_<N>.png index
        """
        pattern = re.compile(r"franka_image_(\d+)\.png")  #for heal use 'png' , for franka use 'franka_image_(\d+)\.png'
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

        self.start_index = self._get_start_index()
        self.saved_count = 0
        self.max_images = 1

        self.get_logger().info(f"Starting image index at {self.start_index}")
        self.get_logger().info(f"Will save {self.max_images} image(s)")

        # Thread-safe queue
        self.image_queue = queue.Queue()

        # Background writer thread
        self.writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True
        )
        self.writer_thread.start()

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        # Timer to safely shutdown
        self.shutdown_timer = self.create_timer(
            0.2, self._check_shutdown
        )

    def image_callback(self, msg: Image):
        """
        FAST callback: no blocking work here
        """
        if self.saved_count >= self.max_images:
            return

        self.image_queue.put(msg)
        self.saved_count += 1

    def _writer_loop(self):
        """
        Background thread: safe to do OpenCV + disk I/O
        """
        while rclpy.ok():
            try:
                msg = self.image_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                cv_image = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='bgr8'
                )

                file_index = self.start_index
                filename = f"franka_image_{file_index}.png"
                filepath = os.path.join(self.save_dir, filename)

                cv2.imwrite(filepath, cv_image)

                self.get_logger().info(
                    f"Saved 1/{self.max_images}: {filepath}"
                )

            except Exception as e:
                self.get_logger().error(f"Failed to save image: {e}")

    def _check_shutdown(self):
        """
        Clean shutdown outside callbacks
        """
        if self.saved_count >= self.max_images:
            self.get_logger().info("Saved required images. Shutting down safely.")
            self.destroy_timer(self.shutdown_timer)
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
