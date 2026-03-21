import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import os
import re   
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
#  joint angles for Franka for calibration

DT = 0.005 # 1 kHz

class HomeFR3Commander(Node):
    def __init__(self):
        super().__init__("Franka_Calibration")

        self.joint_trajectory = np.array([[-0.00962888, -0.0046116, 0.05383474, -1.86224878, -0.28318989, 1.61535454, 0.47778386],[-0.34081504, 0.47660032, 0.10504995, -1.43124712, -0.62942487, 1.62142432, 0.12660879],[-0.36311445, 0.87897891, 0.11122602, -1.37256217, -0.47405246, 2.02747297, 0.12725411],[-0.77647948, 0.76503295, 0.21153893, -1.47480702, -0.79653788, 1.82421291, 0.12696777],[-1.36400211, 0.42377692, 0.42194545, -2.27365685, -0.57426643, 3.32494879, 0.88902348],[-0.90047228, -0.12269092, -0.15039368, -2.43888378, -0.13313277, 2.70000625, 0.80804646],[-1.38673782, -0.24599674, -0.01297847, -2.4409883, -0.59954143, 2.70004559, 0.80803758],[-0.77442056, 0.84539735, -0.16944727, -1.64400172, -0.04002491, 3.0203371, 0.97803539],[-0.72138393, 0.99947053, 0.04701513, -1.27299201, -0.04722387, 3.29181147, 0.8713237],[-0.49559259, 0.82619548, 0.16646703, -1.49736273, -0.60066974, 2.14133883, 0.33042184],[-0.50372666, 0.51345408, 0.70526528, -1.48606336, -0.8490175, 1.73019469, 0.33046803],[-0.80257869, 0.17781396, 0.75609654, -1.35453999, -0.85443139, 1.49656916, 0.32596052],[-0.71557462, 0.22441283, 1.18846786, -1.70084739, -0.8519209, 1.49661982, 0.3259677],[0.13835181, 0.78501964, 0.43012798, -1.49801362, -1.18758726, 1.65342546, 0.7805447],[-0.18682173, 0.46857369, 0.52834684, -1.93775845, -1.79282176, 1.29987729, 0.1526491],[-0.25479388, 0.01068015, 0.63753301, -2.753968, -1.77673352, 1.53139806, 0.15266068],[-0.18088685, 0.69938302, 0.92355227, -2.13819885, -2.21497774, 1.41069055, 0.15265137],[-0.86724395, -0.30724648, 0.70087218, -2.24536633, -2.21298409, 1.41501725, 0.15265778],[-0.57802409, 0.26346356, -0.55400443, -2.09447765, 2.03311014, 1.78364384, 0.78363037],[-0.578031, 0.00657997, -0.49560672, -1.53700042, 2.03310919, 1.78373098, 0.88438469],[-0.57878524, -0.01174502, -0.27400562, -1.85521913, 2.03218865, 1.78567076, 1.49984777],[-0.95070881, 0.62013221, -0.29035929, -1.86456203, 2.03219438, 1.7861166, 1.49983609],[-0.94549698, 0.22352186, 0.88532346, -2.11686707, 0.99848503, 3.61861944, 0.12691122],[-0.56750387, -0.06247356, 0.11871853, -1.69913185, -0.71599042, 2.43001437, 2.36310267],[-0.57899123, 0.20196407, 0.19100598, -2.25804257, -1.9528563, 2.18825674, 2.75902605],[-0.57330114, 0.31895739, 0.99017143, -2.08583617, -1.95223403, 2.54383183, 2.7646358],[-1.01458669, 0.33641312, 1.35942113, -2.28648329, -1.4234879, 2.69463205, 2.70279312],[-0.63398421, -0.01726586, 1.28278244, -2.22325134, -1.42267227, 2.08138204, 3.03729105],[-0.65480334, -0.01892406, 1.73928452, -2.54773259, -1.42329514, 2.07941437, 3.04979467],[-0.076566, 0.48914596, 0.22171974, -1.92870176, -1.43682456, 1.94534087, 0.7853936],[-0.34458247, -0.04804268, 0.23619452, -1.55052733, -1.23831081, 2.65622139, -0.59168243],[-1.22708571, 0.04122727, 0.12123588, -1.66803455, -1.51510358, 2.39352036, -0.92843223],[-1.51498318, -0.11286526, 0.12064254, -1.97010839, -1.60620308, 2.39601231, -0.92841202],[-1.48921919, 0.98548931, 0.68432677, -1.49757802, -2.83682179, 1.6082505, -0.39086622],[-1.4165169, 1.0155375, 0.94504356, -1.7093488, -2.82874084, 2.08138227, -0.43431228],[0.43892476, 1.53630817, -1.46026826, -1.76772249, -2.67603374, 2.49458265, 1.79119766],[0.92273003, 1.53474474, -1.47406006, -1.78335857, -2.67603564, 2.49456382, 1.7911967],[1.38100266, 1.53473175, -1.45260561, -1.78412354, -2.67606473, 2.49456143, 1.79119289],[-1.19082761, 0.88697606, 1.39170802, -2.33704352, -2.85473585, 1.6901114, -0.08622412],[-1.76405609, 0.21948691, 1.57816124, -2.01022983, -2.85294652, 1.68913245, -0.08621244],[-1.99785233, 0.15756226, 1.80346525, -2.14405847, -1.56983089, 0.73040038, -0.08620854],[-1.99743211, -0.20628232, 2.88170171, -2.42598128, -1.16808259, 1.60129297, -0.08618013],[0.40526554, -1.40427399, -1.01095903, -2.44121933, -0.05046101, 2.51804328, -0.83783066],[0.25823, -1.46965206, -1.09497213, -2.17262053, -0.0313738, 2.5170157, -0.83783168],[0.23756605, -1.74890423, -1.90139556, -2.76868343, -0.03332275, 2.517205, -0.83766645],[1.33738708, -1.74560702, -2.00453711, -3.02563167, -0.02561974, 2.46831489, -0.88047153],[-1.33740449, 1.16536117, 1.65578616, -1.95495343, 2.33160782, 1.4244802, 0.76553243],[-0.80741644, 0.27818137, 1.16178334, -1.86731768, 1.67194986, 3.32081795, -0.27662787],[-0.29329145, -0.25861391, 1.42402041, -2.49697137, -0.91943395, 1.60032082, 0.69507378]], dtype=float)
        self.home_joints = np.array([-0.00039212, -0.78243261, 0.00056831, -2.35548115, -0.00093033, 1.57301354, 0.78430986], dtype=float)
        self.state = "MOVE_TO_TARGET"
        self.hold_start_time = None
        self.current_index = 0
        assert self.joint_trajectory.shape[1] == 7
        self.bridge = CvBridge()
        self.rgb_msg = None
        self.depth_msg = None
        self.save_dir = "/home/ubuntu/Deepak_WS/Roboreg-IITGN/Roboreg_test/franka_side"
        os.makedirs(self.save_dir, exist_ok=True)
        self.image_index = self.get_start_index()
        self.last_rgb_time = None
        self.last_depth_time = None
        self.frame_timeout = 1.0 

        # Controller gains
        self.Kp = 1.5
        self.max_vel = 0.1
        self.goal_tolerance = 0.01

        #Publisher
        self.pub = self.create_publisher(
            Float64MultiArray,
            "/NS_1/joint_velocity_controller/commands",
            10
        )

        # Subscriber to FR3 joint states
        self.sub = self.create_subscription(
            JointState,
            "/NS_1/joint_states",
            self.joint_state_callback,
            10
        )
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.rgb_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        self.current_joints = None
        self.dt = DT
        self.timer = self.create_timer(self.dt, self.control_loop)

    def joint_state_callback(self, msg: JointState):
        # read first 7 joints
        if len(msg.position) < 7:
            self.get_logger().warn("Invalid joint points")
            return
        self.current_joints = np.array(msg.position[:7])

    def rgb_callback(self, msg):
        self.rgb_msg = msg
        self.last_rgb_time = self.get_clock().now().nanoseconds * 1e-9

    def depth_callback(self, msg):
        self.depth_msg = msg
        self.last_depth_time = self.get_clock().now().nanoseconds * 1e-9

    def check_frame_freshness(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_rgb_time is None or self.last_depth_time is None:
            raise RuntimeError("No camera frames received yet")
        
        if self.last_rgb_time < self.capture_start_time:
            raise RuntimeError("RGB frame is stale (before pose reached)")

        if self.last_depth_time < self.capture_start_time:
            raise RuntimeError("Depth frame is stale (before pose reached)")

        if now - self.last_rgb_time > self.frame_timeout:
            raise RuntimeError("RGB stream timeout")

        if now - self.last_depth_time > self.frame_timeout:
            raise RuntimeError("Depth stream timeout")
    
    def get_start_index(self):
        pattern = re.compile(r"franka_image_(\d+)\.png")
        max_index = -1
        for fname in os.listdir(self.save_dir):
            match = pattern.match(fname)
            if match:
                idx = int(match.group(1))
                max_index = max(max_index, idx)
        return max_index + 1

    def control_loop(self):
        if self.current_joints is None:
            return

        if self.state == "DONE":
            msg = Float64MultiArray()
            msg.data = np.zeros(7).tolist()
            self.pub.publish(msg)
            return
        
        elif self.state == "MOVE_TO_TARGET":
            target = self.joint_trajectory[self.current_index]
        elif self.state == "MOVE_TO_HOME":
            target = self.home_joints
        
        elif self.state == "CAPTURE":
            
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
                return
            elif time.time() - self.hold_start_time < 1.0:
                return
            
            try:
                self.check_frame_freshness()
            except RuntimeError as e:
                self.get_logger().error(str(e))
                return

            try:
                # --- RGB ---
                cv_image = self.bridge.imgmsg_to_cv2(
                    self.rgb_msg, desired_encoding='bgr8'
                )

                rgb_filename = f"franka_image_{self.image_index}.png"
                rgb_path = os.path.join(self.save_dir, rgb_filename)
                cv2.imwrite(rgb_path, cv_image)

                # --- DEPTH ---
                depth_mm = np.frombuffer(
                    self.depth_msg.data,
                    dtype=np.uint16
                ).reshape(self.depth_msg.height, self.depth_msg.width)
                depth_m = depth_mm.astype(np.float32) * 0.001
                depth_filename = f"franka_depth_{self.image_index}.npy"
                depth_path = os.path.join(self.save_dir, depth_filename)
                np.save(depth_path, depth_m)
                self.get_logger().info(f"Saved RGB + Depth index {self.image_index}")
                self.image_index += 1
                
                # Move back
                self.state = "MOVE_TO_HOME"

            except Exception as e:
                self.get_logger().error(f"Capture failed: {e}")

            # stop motion during capture
            self.hold_start_time = None
            msg = Float64MultiArray()
            msg.data = np.zeros(7).tolist()
            self.pub.publish(msg)
            return
        
        else:
            vel_cmd = np.zeros(7)
            msg = Float64MultiArray()
            msg.data = vel_cmd.tolist()
            self.pub.publish(msg)
            return

        #Motion Control
        error = target - self.current_joints
        error_norm = np.linalg.norm(error)

        if error_norm < self.goal_tolerance:
            vel_cmd = np.zeros(7)

            if self.state == "MOVE_TO_TARGET":
                self.get_logger().info("Reached target joint configuration")
                self.state = "CAPTURE"
                self.hold_start_time = None
                self.capture_start_time = self.get_clock().now().nanoseconds * 1e-9

            elif self.state == "MOVE_TO_HOME":
                self.get_logger().info("Returned to home position")
                self.current_index += 1
                if self.current_index >= len(self.joint_trajectory):
                    self.get_logger().info("All poses completed")
                    self.state = "DONE"
                else:
                    self.state = "MOVE_TO_TARGET"

        else:
            vel_cmd = self.Kp * error
            vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)

        msg = Float64MultiArray()
        msg.data = vel_cmd.tolist()
        self.pub.publish(msg)

if __name__ == "__main__":
    rclpy.init(args=None)
    node = HomeFR3Commander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()