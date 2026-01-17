#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg
import numpy as np
from tf2_ros import TransformBroadcaster
import math

class TransformationPublisher(Node):

    def __init__(self):
        super().__init__('transformation_publisher')
        
        # Create a TF2 broadcaster
        self.broadcaster = TransformBroadcaster(self)
        
        # Create a timer to publish the transformation every 0.1s
        self.timer = self.create_timer(0.1, self.publish_transform)

    def publish_transform(self):
        # Define the transformation matrix (rotation and translation)
        T = np.array([[-0.8627,  0.3074,  0.4016,  0.8729],
        [-0.3599,  0.1847, -0.9145,  0.7893],
        [-0.3553, -0.9335, -0.0487,  0.6163],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])

        # Extract the rotation matrix (3x3) and translation vector (3x1)
        rotation = T[:3, :3]
        translation = T[:3, 3]

        # Convert the rotation matrix to Euler angles
        roll, pitch, yaw = self.rotation_matrix_to_euler_angles(rotation)

        # Convert Euler angles to quaternion
        quat = self.euler_to_quaternion(roll, pitch, yaw)

        # Create a TransformStamped message
        t = geometry_msgs.msg.TransformStamped()

        # Set the header information (frame_id and child_frame_id)
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'  # Parent frame
        t.child_frame_id = 'target_frame'  # Child frame

        # Set translation (position) and rotation (orientation)
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        # Broadcast the transformation
        self.broadcaster.sendTransform(t)

        self.get_logger().info("Published transform from base_link to target_frame")

    def rotation_matrix_to_euler_angles(self, rotation_matrix):
        """Converts a 3x3 rotation matrix to Euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        return x, y, z

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Converts Euler angles (roll, pitch, yaw) to a quaternion."""
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    node = TransformationPublisher()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
