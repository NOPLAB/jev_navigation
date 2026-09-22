"""One ROS node: camera input, asynchronous inference, direct cmd_vel output."""

import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import cv2
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool

from .client import decide
from .policy import CURVATURES, MotionState, choose_action
from .path_tracking import PathFollower, make_path


class DecisionNode(Node):
    def __init__(self):
        super().__init__("decision_node")
        defaults = {
            "server_url": "http://127.0.0.1:8000",
            "goal": "Approach the red chair and stop before touching it.",
            "dry_run": True,
            "forward_speed": 0.1,
            "turn_speed": 0.25,
            "command_ttl": 0.8,
            "request_timeout": 2.0,
            "inference_interval": 0.2,
            "publish_rate": 20.0,
            "min_probability": 0.2,
            "min_margin": 0.0,
            "image_max_side": 384,
            "path_length": 0.6,
            "lookahead": 0.2,
            "path_tolerance": 0.04,
            "max_path_error": 0.25,
            "linear_acceleration": 0.2,
            "angular_acceleration": 0.8,
            "switch_margin": 0.08,
            "odom_timeout": 0.3,
            "pose_sync_tolerance": 0.1,
            "odom_jump_distance": 0.5,
            "odom_jump_angle": 0.7,
            "odom_frame": "odom",
            "base_frame": "base_link",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value, ParameterDescriptor(read_only=name != "goal"))
        self.cfg = {name: self.get_parameter(name).value for name in defaults}
        for name in ("forward_speed", "turn_speed", "command_ttl", "request_timeout",
                     "inference_interval", "publish_rate", "path_length", "lookahead",
                     "path_tolerance", "max_path_error", "linear_acceleration",
                     "angular_acceleration", "odom_timeout", "pose_sync_tolerance",
                     "odom_jump_distance", "odom_jump_angle"):
            value = self.cfg[name]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_probability", "min_margin", "switch_margin"):
            if not 0 <= self.cfg[name] <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if not 64 <= self.cfg["image_max_side"] <= 1024:
            raise ValueError("image_max_side must be between 64 and 1024")
        if urlparse(self.cfg["server_url"]).scheme not in ("http", "https"):
            raise ValueError("server_url must use HTTP or HTTPS")
        self._check_goal(self.cfg["goal"])
        if not self.cfg["path_tolerance"] < self.cfg["lookahead"] < self.cfg["path_length"]:
            raise ValueError("Require path_tolerance < lookahead < path_length")
        if self.cfg["path_length"] > 0.7:
            raise ValueError("path_length must be <= 0.7 m for the fixed forward arcs")
        if not self.cfg["odom_frame"] or not self.cfg["base_frame"]:
            raise ValueError("Odometry frame names must not be empty")
        self.motion = MotionState()
        self.follower = PathFollower()
        self.odom = deque(maxlen=200)
        self.last_publish = time.monotonic()
        self.bridge = CvBridge()
        self.latest = None
        self.future = None
        self.sequence = 0
        self.next_inference = 0.0
        self.previous_ros_time = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.status_pub = self.create_publisher(DiagnosticArray, "/jev/status", 1)
        self.path_pub = self.create_publisher(Path, "/jev/path", 1)
        self.create_subscription(Image, "/camera/image_raw", self.on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        self.create_service(SetBool, "/jev/enable", self.on_enable)
        self.add_on_set_parameters_callback(self.on_parameters)
        # A steady timer continues stopping the base even if simulated ROS time pauses.
        self.timer = self.create_timer(1.0 / self.cfg["publish_rate"], self.tick,
                                      clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info("Ready, disabled. dry_run=" + str(self.cfg["dry_run"]))

    @staticmethod
    def _check_goal(goal):
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 512:
            raise ValueError("goal must be a nonempty English instruction of at most 512 characters")

    def on_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name == "goal":
                if self.motion.enabled:
                    return SetParametersResult(successful=False, reason="Disable before changing goal")
                try:
                    self._check_goal(parameter.value)
                except ValueError as exc:
                    return SetParametersResult(successful=False, reason=str(exc))
        return SetParametersResult(successful=True)

    def on_enable(self, request, response):
        if request.data and not self.odom_fresh():
            response.success = False
            response.message = "Fresh valid odometry required before enabling"
            return response
        self.motion.enable(request.data)
        self.clear_path()
        self.latest = None  # Require a new camera frame after every enable/disable.
        self.publish_velocity()
        response.success = True
        response.message = "enabled" if request.data else "disabled"
        self.report(response.message)
        return response

    def on_image(self, message):
        if self.motion.enabled:
            self.latest = (message, time.monotonic())

    def odom_fresh(self):
        if not self.odom:
            return False
        stamp, received, _ = self.odom[-1]
        age = self.get_clock().now().nanoseconds / 1e9 - stamp
        return (-0.05 <= age < self.cfg["odom_timeout"]
                and time.monotonic() - received < self.cfg["odom_timeout"])

    def odom_fault(self, reason):
        self.odom.clear()
        self.motion.enable(False)  # Invalidate pending inference; explicit re-enable required.
        self.latest = None
        self.clear_path()
        self.publish_velocity()
        self.report(reason)

    def on_odom(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        age = self.get_clock().now().nanoseconds / 1e9 - stamp
        norm = sum(v * v for v in (q.x, q.y, q.z, q.w))
        if (message.header.frame_id != self.cfg["odom_frame"]
                or message.child_frame_id != self.cfg["base_frame"]
                or not all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
                or not 0.99 <= norm <= 1.01 or stamp <= 0
                or not -0.05 <= age < self.cfg["odom_timeout"]):
            self.odom_fault("invalid odometry; disabled")
            return
        pose = (p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y),
                                    1 - 2 * (q.y * q.y + q.z * q.z)))
        if self.odom:
            old_stamp, _, old = self.odom[-1]
            angle = abs(math.atan2(math.sin(pose[2] - old[2]), math.cos(pose[2] - old[2])))
            if (stamp < old_stamp or math.dist(pose[:2], old[:2]) > self.cfg["odom_jump_distance"]
                    or angle > self.cfg["odom_jump_angle"]):
                self.odom_fault("odometry reset/jump; disabled")
                return
            if stamp == old_stamp:
                return  # Replayed samples must not refresh the receive timeout.
        self.odom.append((stamp, time.monotonic(), pose))

    def publish_path(self):
        message = Path()
        message.header.frame_id = self.cfg["odom_frame"]
        message.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.follower.points:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        self.path_pub.publish(message)

    def clear_path(self):
        had_path = bool(self.follower.points)
        self.follower.reset()
        if had_path:
            self.publish_path()

    def report(self, reason, result=None):
        status = DiagnosticStatus(name="jev_navigation", hardware_id="decider-2b-vision",
                                  level=DiagnosticStatus.OK if reason == "decision" else DiagnosticStatus.WARN,
                                  message=reason)
        values = {"enabled": self.motion.enabled, "dry_run": self.cfg["dry_run"],
                  "action": self.motion.action}
        if result:
            values.update(result["probabilities"])
            values["inference_ms"] = result.get("inference_ms", "unknown")
        status.values = [KeyValue(key=k, value=str(v)) for k, v in values.items()]
        message = DiagnosticArray(status=[status])
        message.header.stamp = self.get_clock().now().to_msg()
        self.status_pub.publish(message)

    def publish_velocity(self):
        now = time.monotonic()
        dt, self.last_publish = now - self.last_publish, now
        linear = angular = 0.0
        if self.motion.active(now) and self.odom_fresh():
            target = self.follower.target(self.odom[-1][2], self.cfg["lookahead"],
                         self.cfg["path_tolerance"], self.cfg["max_path_error"],
                         self.cfg["forward_speed"], self.cfg["turn_speed"],
                         self.cfg["linear_acceleration"])
            if target is None:
                self.motion.stop()
                self.clear_path()
                self.report("path completed or untrackable")
            elif not self.cfg["dry_run"]:
                linear, angular = self.follower.slew(target, dt,
                    self.cfg["linear_acceleration"], self.cfg["angular_acceleration"])
        else:
            self.clear_path()
        message = Twist()
        message.linear.x, message.angular.z = linear, angular
        self.cmd_pub.publish(message)

    def infer(self, frame, goal, previous_action, request_id):
        image = self.bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8")
        height, width = image.shape[:2]
        scale = min(1.0, self.cfg["image_max_side"] / max(height, width))
        if scale < 1:
            image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return decide(self.cfg["server_url"], encoded.tobytes(), goal, previous_action,
                      request_id, self.cfg["request_timeout"])

    def tick(self):
        now = time.monotonic()
        ros_now = self.get_clock().now().nanoseconds / 1e9
        if self.previous_ros_time is not None and ros_now < self.previous_ros_time:
            self.odom_fault("ROS clock moved backwards; disabled")
        self.previous_ros_time = ros_now
        if self.motion.enabled and not self.odom_fresh():
            self.odom_fault("odometry stale; disabled")
        if self.future is not None and self.future.done():
            future, generation, deadline = self.future, self.pending_generation, self.pending_deadline
            self.future = None
            try:
                result = future.result()
                if generation == self.motion.generation and self.motion.enabled:
                    action = choose_action(result["probabilities"], self.cfg["min_probability"],
                                           self.cfg["min_margin"], self.motion.action,
                                           self.cfg["switch_margin"])
                    previous = self.motion.action
                    accepted = self.motion.accept(action, generation, deadline, now)
                    if accepted and action in CURVATURES:
                        points = self.follower.points
                        # Keep the odom-anchored path while it has useful distance left.
                        remaining = sum(math.dist(a, b) for a, b in
                                        zip(points[self.follower.index:], points[self.follower.index + 1:]))
                        if action != previous or remaining < 2 * self.cfg["lookahead"]:
                            self.follower.set_path(make_path(action, self.pending_pose, self.cfg["path_length"]))
                            self.publish_path()
                    else:
                        self.clear_path()
                    self.report("goal_reached" if accepted and action == "goal_reached"
                                else "decision" if accepted else "expired inference", result)
            except Exception as exc:
                if generation == self.motion.generation:
                    self.motion.stop()
                    self.report("inference failed: " + str(exc))
                    self.get_logger().warning(str(exc))
        if self.motion.enabled and self.future is None and self.latest and now >= self.next_inference:
            frame, received = self.latest
            self.latest = None
            stamp = frame.header.stamp.sec + frame.header.stamp.nanosec / 1e9
            age = ros_now - stamp
            deadline = min(received + self.cfg["command_ttl"],
                           now + self.cfg["command_ttl"] - max(0.0, age))
            if stamp <= 0 or age < -0.05 or now >= deadline:
                self.motion.stop()
                self.report("invalid or stale camera timestamp")
            else:
                sample = min(self.odom, key=lambda item: abs(item[0] - stamp))
                if abs(sample[0] - stamp) > self.cfg["pose_sync_tolerance"]:
                    self.motion.stop()
                    self.clear_path()
                    self.report("no odometry aligned with camera timestamp")
                    self.publish_velocity()
                    return
                self.sequence += 1
                self.pending_pose = sample[2]
                self.pending_generation = self.motion.generation
                self.pending_deadline = deadline
                self.next_inference = now + self.cfg["inference_interval"]
                self.future = self.pool.submit(self.infer, frame, self.get_parameter("goal").value,
                                               self.motion.action, self.sequence)
        expired = self.motion.action in CURVATURES and now >= self.motion.deadline
        self.publish_velocity()
        if expired:
            self.report("command expired")

    def close(self):
        self.motion.enable(False)
        self.publish_velocity()
        self.pool.shutdown(wait=False, cancel_futures=True)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DecisionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.close()
            else:
                node.pool.shutdown(wait=False, cancel_futures=True)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
