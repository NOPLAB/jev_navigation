"""One ROS node: camera input, asynchronous inference, direct cmd_vel output."""

import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import cv2
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool

from .client import decide
from .policy import MotionState, choose_action


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
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value, ParameterDescriptor(read_only=name != "goal"))
        self.cfg = {name: self.get_parameter(name).value for name in defaults}
        import math
        for name in ("forward_speed", "turn_speed", "command_ttl", "request_timeout",
                     "inference_interval", "publish_rate"):
            value = self.cfg[name]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_probability", "min_margin"):
            if not 0 <= self.cfg[name] <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if not 64 <= self.cfg["image_max_side"] <= 1024:
            raise ValueError("image_max_side must be between 64 and 1024")
        if urlparse(self.cfg["server_url"]).scheme not in ("http", "https"):
            raise ValueError("server_url must use HTTP or HTTPS")
        self._check_goal(self.cfg["goal"])
        self.motion = MotionState()
        self.bridge = CvBridge()
        self.latest = None
        self.future = None
        self.sequence = 0
        self.next_inference = 0.0
        self.previous_ros_time = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.status_pub = self.create_publisher(DiagnosticArray, "/jev/status", 1)
        self.create_subscription(Image, "/camera/image_raw", self.on_image, qos_profile_sensor_data)
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
        self.motion.enable(request.data)
        self.latest = None  # Require a new camera frame after every enable/disable.
        self.publish_velocity()
        response.success = True
        response.message = "enabled" if request.data else "disabled"
        self.report(response.message)
        return response

    def on_image(self, message):
        if self.motion.enabled:
            self.latest = (message, time.monotonic())

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
        linear, angular = self.motion.velocity(time.monotonic(), self.cfg["forward_speed"],
                                              self.cfg["turn_speed"], self.cfg["dry_run"])
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
            self.motion.enable(False)
            self.latest = None
            self.report("ROS clock moved backwards; disabled")
        self.previous_ros_time = ros_now
        if self.future is not None and self.future.done():
            future, generation, deadline = self.future, self.pending_generation, self.pending_deadline
            self.future = None
            try:
                result = future.result()
                if generation == self.motion.generation and self.motion.enabled:
                    action = choose_action(result["probabilities"], self.cfg["min_probability"],
                                           self.cfg["min_margin"])
                    accepted = self.motion.accept(action, generation, deadline, now)
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
                self.sequence += 1
                self.pending_generation = self.motion.generation
                self.pending_deadline = deadline
                self.next_inference = now + self.cfg["inference_interval"]
                self.future = self.pool.submit(self.infer, frame, self.get_parameter("goal").value,
                                               self.motion.action, self.sequence)
        was_moving = self.motion.action not in ("stop", "goal_reached")
        self.publish_velocity()
        if was_moving and self.motion.action == "stop":
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
