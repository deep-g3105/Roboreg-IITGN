import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import os
import re
import time

from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge


DT = 0.01  # 100 Hz


class HealJointCollector(Node):

    def __init__(self):
        super().__init__("Heal_Calibration")

        # JOINT TRAJECTORY 
        pose = np.load(f"heal_joints_0.npy")
        if pose.shape == (6,):
            pose = pose.reshape(1, -1)

        self.joint_trajectory = pose

        self.home_joints = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        assert self.joint_trajectory.shape[1] == 6
        self.state = "MOVE_TO_TARGET"
        self.current_index = 0
        self.hold_start_time = None
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self._name_to_idx = {}
        
        #Controller Gains
        self.Kp = 1.5
        self.max_vel = 0.1
        self.goal_tolerance = 0.01
        
        # Camera
        self.bridge = CvBridge()
        self.rgb_msg = None
        self.depth_msg = None

        self.last_rgb_time = None
        self.last_depth_time = None
        self.frame_timeout = 1.0
        self.save_dir = os.path.expanduser("~/heal_top_calib")
        os.makedirs(self.save_dir, exist_ok=True)
        self.image_index = self.get_start_index()

        # Joint states
        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

        # Velocity command
        self.pub = self.create_publisher(
            Float64MultiArray,
            "/velocity_controller/commands",
            10
        )

        # Camera
        self.rgb_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            10
        )

        self.current_joints = None
        self.timer = self.create_timer(DT, self.control_loop)
        self.get_logger().info("HEAL Joint Collector Started")

    def joint_state_callback(self, msg):
        if not msg.name or not msg.position:
            return

        # Build name → index map
        self._name_to_idx = {n: i for i, n in enumerate(msg.name)}

        q = np.zeros(6, dtype=float)
        missing = []

        for i, name in enumerate(self.joint_names):
            idx = self._name_to_idx.get(name, None)
            if idx is None:
                missing.append(name)
                q[i] = 0.0
            else:
                q[i] = float(msg.position[idx])

        if missing:
            self.get_logger().warn(f"Missing joints: {missing}")

        self.current_joints = q

    def rgb_callback(self, msg):
        self.rgb_msg = msg
        self.last_rgb_time = self.get_clock().now().nanoseconds * 1e-9

    def depth_callback(self, msg):
        self.depth_msg = msg
        self.last_depth_time = self.get_clock().now().nanoseconds * 1e-9

    def check_frame_freshness(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_rgb_time is None or self.last_depth_time is None:
            raise RuntimeError("No frames received")

        if self.last_rgb_time < self.capture_start_time:
            raise RuntimeError("Stale RGB")

        if self.last_depth_time < self.capture_start_time:
            raise RuntimeError("Stale Depth")

        if now - self.last_rgb_time > self.frame_timeout:
            raise RuntimeError("RGB timeout")

        if now - self.last_depth_time > self.frame_timeout:
            raise RuntimeError("Depth timeout")

    def get_start_index(self):
        pattern = re.compile(r"heal_image_(\d+)\.png")
        max_idx = -1
        for f in os.listdir(self.save_dir):
            m = pattern.match(f)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def control_loop(self):

        if self.current_joints is None:
            return

        if self.state == "DONE":
            self.publish_zero()
            return

        elif self.state == "MOVE_TO_TARGET":
            target = self.joint_trajectory[self.current_index]
            self.get_logger().info(f"Moving to target: {target}")

        elif self.state == "CAPTURE":

            self.publish_zero()

            if self.hold_start_time is None:
                self.hold_start_time = time.time()
                return

            if time.time() - self.hold_start_time < 1.0:
                return

            try:
                self.check_frame_freshness()
                rgb = self.bridge.imgmsg_to_cv2(self.rgb_msg, "bgr8")
                depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
                depth = depth.astype(np.float32) * 0.001
                cv2.imwrite(
                    os.path.join(self.save_dir, f"heal_image_{self.image_index}.png"),
                    rgb
                )
                np.save(
                    os.path.join(self.save_dir, f"heal_depth_{self.image_index}.npy"),
                    depth
                )
                np.save(
                    os.path.join(self.save_dir, f"heal_joints_{self.image_index}.npy"),
                    self.current_joints
                )
                self.get_logger().info(f"Saved index {self.image_index}")
                self.image_index += 1

                self.state = "DONE"
                self.get_logger().info("DONE. Ready for next pose.")

            except Exception as e:
                self.get_logger().error(f"Capture failed: {e}")

            self.hold_start_time = None
            return

        else:
            self.publish_zero()
            return

        error = target - self.current_joints
        error_norm = np.linalg.norm(error)

        if error_norm < self.goal_tolerance:

            self.publish_zero()

            if self.state == "MOVE_TO_TARGET":
                self.state = "CAPTURE"
                self.capture_start_time = self.get_clock().now().nanoseconds * 1e-9

        else:
            vel = self.Kp * error
            vel = np.clip(vel, -self.max_vel, self.max_vel)

            msg = Float64MultiArray()
            msg.data = vel.tolist()
            self.pub.publish(msg)

    def publish_zero(self):
        msg = Float64MultiArray()
        msg.data = np.zeros(6).tolist()
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = HealJointCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()