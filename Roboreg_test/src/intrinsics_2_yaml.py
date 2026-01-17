#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import yaml


class CameraInfoSaver(Node):

    def __init__(self):
        super().__init__('camera_info_saver')

        self.subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.callback,
            10
        )

        self.saved = False

    def callback(self, msg):
        if self.saved:
            return

        camera_info_dict = {
            "header": {
                "stamp": {
                    "sec": int(msg.header.stamp.sec),
                    "nanosec": int(msg.header.stamp.nanosec)
                },
                "frame_id": msg.header.frame_id
            },
            "height": int(msg.height),
            "width": int(msg.width),
            "distortion_model": msg.distortion_model,
            "d": [float(x) for x in msg.d],
            "k": [float(x) for x in msg.k],
            "r": [float(x) for x in msg.r],
            "p": [float(x) for x in msg.p],
            "binning_x": int(msg.binning_x),
            "binning_y": int(msg.binning_y),
            "roi": {
                "x_offset": int(msg.roi.x_offset),
                "y_offset": int(msg.roi.y_offset),
                "height": int(msg.roi.height),
                "width": int(msg.roi.width),
                "do_rectify": bool(msg.roi.do_rectify)
            }
        }

        with open("camera_side_info.yaml", "w") as f:
            yaml.safe_dump(camera_info_dict, f, sort_keys=False)

        self.get_logger().info("Camera info saved to camera_side_info.yaml")
        self.saved = True


def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
