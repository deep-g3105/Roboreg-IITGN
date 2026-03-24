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
        self.joint_trajectory = np.array([
        [0.217823, 0.120596, 0.0111876, -0.049877, 0.147858, 0.0507394],
        [0.24239, 0.211498, 0.0805171, 0.163753, 0.176582, 0.0507514],
        [-0.278504, 0.21151, 0.0805291, 0.163681, 0.394105, 0.0507514],
        [-0.185673, 0.1667, 0.088267, 0.163645, 0.451289, 0.182248],
        [-0.312797, 0.535819, 0.0716293, 0.441107, 0.338311, -0.358518],
        [-0.107708, 0.474095, -0.264837, 0.441143, 0.338347, -0.358542],
        [-0.107731, -0.482624, -0.264993, 0.441095, 0.338239, -0.35853],
        [-0.0187578, 0.408647, -0.280983, 0.0547162, 0.710137, -0.432759],
        #[-0.0804093, 0.454211, 0.260082, -0.167718, 0.710125, -0.432759],
        [-0.0804093, 0.801051, 0.260177, -0.167694, 0.710173, -0.432759],
        [0.252595, 0.584019, 0.371179, -0.16773, 0.710125, -0.432771],
        [0.285655, 1.04342, 0.575814, -0.167694, 0.710161, -0.432747],
        [-0.218805, 0.895822, 0.559487, -0.167706, 0.710149, -0.432759],
        [-0.218793, 0.827007, 0.559463, -0.167706, 0.710125, -0.432759],
        [-0.15913, 0.622313, 0.441251, -0.167706, 0.710125, -0.432747],
        [0.476299, 0.63913, 0.429189, -0.167754, 0.93081, -0.432747],
        [0.476299, 1.06186, 0.499609, -0.16773, 0.930834, -0.432771],
        [0.238389, 0.62369, 0.499513, -0.167754, 0.930786, -0.432747],
        [0.0670896, 0.697272, -0.0287476, -0.167706, 0.930846, -0.432747],
        [0.653432, 0.651923, -0.0774866, -0.195795, 0.93087, -0.432747],
        [0.653444, 1.05728, 0.578533, -0.195867, 0.930846, -0.432771],
        [0.389649, 1.05729, 0.578545, -0.195867, 0.930834, -0.432759],
        [0.389649, 0.728343, 0.578449, -0.195891, 0.930822, -0.432747],
        [0.0777861, 0.728343, 0.578449, -0.195903, 0.93081, -0.432759],
        [0.0777741, 1.06621, 0.653157, -0.195879, 0.930846, -0.432759],
        [-0.341198, 1.06622, 0.628949, -0.195867, 0.930834, -0.432747],
        [-0.329195, 0.816455, 0.235478, -0.195855, 0.930834, -0.432759],
        [0.00283882, 0.828672, 0.292938, -0.195855, 0.930846, -0.432747],
        [-0.394285, 0.844999, 0.466729, 0.318116, 0.93081, -0.432759],
        [-0.394261, 1.23089, 0.46686, 0.318188, 0.930846, -0.432759],
        [-0.394273, 0.84065, 0.183278, 0.149176, 0.893498, -0.432759],
        [0.0198358, 0.717731, 0.0931541, 0.149164, 0.795038, -0.813927],
        [-0.00019165, 0.524691, 0.0931181, -0.285379, 0.79505, -0.813939],
        [0.642652, 0.943687, 0.624888, -0.379456, 0.795062, -0.813915],
        [-0.211079, 0.835955, 0.624852, -0.379456, 0.79505, -0.813915],
        [0.161322, 0.807555, 0.632338, -0.416408, 0.795074, -0.813927],
        [0.661913, 0.807459, 0.175624, -0.416372, 0.795122, -0.813927],
        [0.745053, 1.01939, 0.68872, -0.416408, 0.795086, -0.813915],
        [0.745065, 0.113373, 0.328129, -0.416408, 0.79505, -0.813927],
        [0.745065, 0.136287, 0.328141, -0.41642, 0.795038, -0.813915],
        [0.745077, 0.571717, 0.451073, -0.416432, 0.795062, -0.813927],
        [0.114307, 0.489643, -0.012613, -0.416408, 0.795098, -0.813927],
        [0.0368448, 0.489643, -0.012625, -2.07115, 0.795433, -0.850353],
        [0.0368328, 0.482324, -0.0126609, -2.07115, 0.882346, -0.793025],
        [0.102461, 0.482312, 0.0125531, -0.329603, 0.882011, -0.793013],
        [0.22901, 0.333484, -0.131089, -0.624289, 0.882035, -0.793025],
        [-0.0901835, 0.333472, -0.172138, 0.0347486, 0.881939, -0.793013],
        [0.0487391, 0.222458, 0.188512, 0.0348325, 0.446797, -0.793013],
        [-0.189051, 0.878921, 0.515935, -0.0880154, 0.446845, -0.793025],
        [0.290554, 0.968314, 0.790163, -0.0880394, 0.446833, -0.793013]
        ], dtype=float)

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
        pattern = re.compile(r"image_(\d+)\.png")
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

        elif self.state == "MOVE_TO_HOME":
            target = self.home_joints

        elif self.state == "CAPTURE":

            self.publish_zero()

            if self.hold_start_time is None:
                self.hold_start_time = time.time()
                return

            if time.time() - self.hold_start_time < 1.0:
                return

            try:
                self.check_frame_freshness()

                # RGB
                rgb = self.bridge.imgmsg_to_cv2(self.rgb_msg, "bgr8")
                cv2.imwrite(
                    os.path.join(self.save_dir, f"image_{self.image_index}.png"),
                    rgb
                )

                # Depth
                depth = np.frombuffer(
                    self.depth_msg.data, dtype=np.uint16
                ).reshape(self.depth_msg.height, self.depth_msg.width)

                depth = depth.astype(np.float32) * 0.001
                np.save(
                    os.path.join(self.save_dir, f"depth_{self.image_index}.npy"),
                    depth
                )

                # Save joints
                np.save(
                    os.path.join(self.save_dir, f"joints_{self.image_index}.npy"),
                    self.current_joints
                )

                self.get_logger().info(f"Saved index {self.image_index}")
                self.image_index += 1

                self.state = "MOVE_TO_HOME"

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

            elif self.state == "MOVE_TO_HOME":
                self.current_index += 1

                if self.current_index >= len(self.joint_trajectory):
                    self.state = "DONE"
                else:
                    self.state = "MOVE_TO_TARGET"

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