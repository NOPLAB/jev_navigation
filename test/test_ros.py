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
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_srvs.srv import SetBool
        from jev_navigation.decision_node import DecisionNode
        from jev_navigation.policy import ACTIONS

        class Handler(BaseHTTPRequestHandler):
            delay = 0.0

            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                time.sleep(Handler.delay)
                body = json.dumps({"request_id": request["request_id"],
                                   "probabilities": {a: float(a == "forward") for a in ACTIONS}}).encode()
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
                    "-r", "/jev/enable:=/jev_test/enable", "-r", "/jev/status:=/jev_test/status"])
        node = DecisionNode()
        probe = Node("jev_test_probe")
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(probe)
        received = []
        probe.create_subscription(Twist, "/jev_test/cmd_vel",
                                  lambda msg: received.append(msg.linear.x), 10)
        camera = probe.create_publisher(Image, "/jev_test/image", 10)
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
