"""Opt-in real ROS/HTTP test; runs only on isolated test topics and domain."""

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@unittest.skipUnless(os.environ.get("RUN_ROS_TESTS") == "1", "Set RUN_ROS_TESTS=1 in a Humble shell")
class RosTest(unittest.TestCase):
    def test_direct_velocity_and_stale_results(self):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry, Path
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_srvs.srv import SetBool
        from jev_navigation.decision_node import DecisionNode
        from jev_navigation.policy import ACTIONS

        class Handler(BaseHTTPRequestHandler):
            delay = 0.0
            action = "forward"

            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                time.sleep(Handler.delay)
                body = json.dumps({"request_id": request["request_id"],
                                   "probabilities": {a: float(a == Handler.action) for a in ACTIONS}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        rclpy.init(domain_id=173, args=["--ros-args", "-p", "dry_run:=false",
                    "-p", f"server_url:=http://127.0.0.1:{server.server_port}",
                    "-p", "command_ttl:=0.5", "-r", "/cmd_vel:=/jev_test/cmd_vel",
                    "-r", "/camera/image_raw:=/jev_test/image",
                    "-r", "/odom:=/jev_test/odom", "-r", "/jev/path:=/jev_test/path",
                    "-r", "/jev/enable:=/jev_test/enable", "-r", "/jev/status:=/jev_test/status"])
        node = DecisionNode()
        probe = Node("jev_test_probe")
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(probe)
        received = []
        commands, paths = [], []
        probe.create_subscription(Twist, "/jev_test/cmd_vel",
                                  lambda msg: received.append(msg.linear.x), 10)
        probe.create_subscription(Twist, "/jev_test/cmd_vel", commands.append, 100)
        probe.create_subscription(Path, "/jev_test/path", paths.append, 10)
        camera = probe.create_publisher(Image, "/jev_test/image", 10)
        odometry = probe.create_publisher(Odometry, "/jev_test/odom", 10)
        odom_enabled = True
        pose_x = 0.0

        def send_odom():
            if odom_enabled:
                msg = Odometry()
                msg.header.stamp = probe.get_clock().now().to_msg()
                msg.header.frame_id = "odom"
                msg.child_frame_id = "base_link"
                msg.pose.pose.position.x = pose_x
                msg.pose.pose.orientation.w = 1.0
                odometry.publish(msg)

        probe.create_timer(.02, send_odom)
        enable = probe.create_client(SetBool, "/jev_test/enable")

        def spin(seconds):
            until = time.monotonic() + seconds
            while time.monotonic() < until:
                executor.spin_once(timeout_sec=0.01)

        def set_enabled(value):
            future = enable.call_async(SetBool.Request(data=value))
            executor.spin_until_future_complete(future, timeout_sec=2)
            self.assertTrue(future.result().success)

        def frame():
            image = Image(height=16, width=16, encoding="bgr8", step=48,
                          data=bytes([0, 0, 255] * 256))
            image.header.stamp = probe.get_clock().now().to_msg()
            camera.publish(image)

        try:
            self.assertTrue(enable.wait_for_service(timeout_sec=5))
            spin(.3)
            self.assertTrue(received and not any(received))
            set_enabled(True)
            received.clear()
            frame()
            spin(.35)
            self.assertTrue(any(v > 0 for v in received), "No direct forward Twist received")
            spin(.4)
            self.assertEqual(received[-1], 0.0, "Camera loss must expire motion")
            Handler.delay = .7
            received.clear()
            frame()
            spin(.9)
            self.assertFalse(any(received), "Expired inference must not start motion")
            frame()
            spin(.15)
            set_enabled(False)
            set_enabled(True)
            received.clear()
            spin(.8)
            self.assertFalse(any(received), "Pre-enable inference must be discarded")
            Handler.delay = 0
            frame()
            spin(.35)
            self.assertTrue(any(received), "Fresh frame should restore motion")
            self.assertTrue(any(path.poses and path.header.frame_id == "odom" for path in paths))
            Handler.action = "right"
            commands.clear()
            frame()
            spin(.3)
            self.assertTrue(any(c.linear.x > 0 and c.angular.z < 0 for c in commands),
                            "Curved path must move forward and turn simultaneously")
            self.assertTrue(all(c.linear.x > 0 for c in commands),
                            "Normal path switching must not inject zero linear velocity")
            Handler.action = "stop"
            frame()
            spin(.2)
            self.assertEqual(received[-1], 0.0, "Winning stop bypasses acceleration ramp")
            Handler.action = "forward"
            frame()
            spin(.3)
            odom_enabled = False
            spin(.4)
            self.assertFalse(node.motion.enabled, "Odometry loss must disable")
            self.assertEqual(received[-1], 0.0)
            odom_enabled = True
            spin(.1)
            set_enabled(True)
            frame()
            spin(.25)
            pose_x = 2.0
            spin(.1)
            self.assertFalse(node.motion.enabled, "Odometry jump must disable")
            self.assertEqual(received[-1], 0.0)
            set_enabled(False)
            spin(.1)
            self.assertEqual(received[-1], 0.0)
        finally:
            node.close()
            executor.shutdown()
            probe.destroy_node()
            node.destroy_node()
            rclpy.shutdown()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
