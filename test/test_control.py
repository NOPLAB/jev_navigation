"""Run: python -m unittest discover -s test -v. No ROS or GPU required."""

import base64
import io
import unittest

from jev_navigation.policy import ACTIONS, MotionState, choose_action, validate_probabilities
from jev_navigation.path_tracking import PathFollower, make_path


def probabilities(action="forward"):
    return {key: float(key == action) for key in ACTIONS}


class ControlTest(unittest.TestCase):
    def test_command_lifetime_and_restart(self):
        state = MotionState()
        self.assertFalse(state.accept("forward", 0, 10.0, 1.0))
        state.enable(True)
        generation = state.generation
        self.assertTrue(state.accept("forward", generation, 2.0, 1.0))
        self.assertTrue(state.active(1.1))
        self.assertTrue(state.active(1.2))
        self.assertEqual(state.deadline, 2.0)  # Publishing never renews the lease.
        self.assertFalse(state.active(2.0))
        state.enable(False)
        state.enable(True)
        self.assertFalse(state.accept("left", generation, 5.0, 2.1))
        self.assertFalse(state.active(2.2))
        self.assertFalse(state.accept("forward", state.generation, 2.0, 2.1))
        self.assertTrue(state.accept("right", state.generation, 3.0, 2.1))
        self.assertTrue(state.active(2.2))
        self.assertTrue(state.accept("goal_reached", state.generation, 3.0, 2.2))
        self.assertFalse(state.enabled)
        self.assertFalse(state.active(2.3))

    def test_uncertain_and_invalid_outputs(self):
        values = dict.fromkeys(ACTIONS, 0.0)
        values.update(forward=.2, gentle_left=.2, gentle_right=.2, left=.2, right=.2)
        self.assertEqual(choose_action(values, .2, 0.0), "forward")
        values.update(forward=.28, gentle_left=.0, gentle_right=.0, left=.17,
                      right=.21, stop=.26, goal_reached=.08)
        self.assertEqual(choose_action(values, .2, 0.0), "forward")
        values.update(forward=.26, stop=.28)
        self.assertEqual(choose_action(values, .2, 0.0, "forward", .08), "stop")
        self.assertEqual(choose_action(probabilities(), .6, .15), "forward")
        self.assertEqual(choose_action(dict.fromkeys(ACTIONS, 1 / len(ACTIONS)), .6, .15), "stop")
        values = dict.fromkeys(ACTIONS, 0.0)
        values.update(forward=.51, left=.49)
        self.assertEqual(choose_action(values, .5, .15), "stop")
        values.update(forward=.49, left=.51)
        self.assertEqual(choose_action(values, .2, 0.0, "forward", .08), "forward")
        values.update(forward=.4, left=.6)
        self.assertEqual(choose_action(values, .2, 0.0, "forward", .08), "left")
        for bad in (float("nan"), float("inf"), -1, 2, True, "1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_probabilities({**probabilities(), "forward": bad})
        for values in ({}, dict.fromkeys(ACTIONS, .1), {**probabilities(), "extra": 0}):
            with self.assertRaises(ValueError):
                validate_probabilities(values)

    def test_paths_tracking_and_ramps(self):
        follower = PathFollower()
        for action, sign in (("forward", 0), ("gentle_left", 1), ("right", -1)):
            follower.set_path(make_path(action, (0, 0, 0), .6))
            target = follower.target((0, 0, 0), .2, .04, .25, .1, .25, .2)
            self.assertGreater(target[0], 0)
            self.assertAlmostEqual(target[1], sign * (.2 if action == "right" else .1) if sign else 0)
            ramp = follower.slew(target, .05, .2, .8)
            self.assertGreater(ramp[0], 0)
            self.assertLessEqual(abs(ramp[1]), .25)
        self.assertAlmostEqual(follower.linear, .03)
        self.assertIsNone(follower.target((10, 0, 0), .2, .04, .25, .1, .25, .2))
        follower.set_path(make_path("forward", (1, 2, 0), .6))
        self.assertIsNone(follower.target((1.6, 2, 0), .2, .04, .25, .1, .25, .2))
        self.assertIsNone(follower.target((1, 2, 3.141592653589793), .2, .04, .25, .1, .25, .2))
        follower.reset()
        self.assertEqual((follower.linear, follower.angular, follower.points), (0, 0, []))
        rotated = make_path("forward", (1, 2, 3.141592653589793 / 2), .6)
        self.assertAlmostEqual(rotated[-1][0], 1)
        self.assertAlmostEqual(rotated[-1][1], 2.6)


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image
        stream = io.BytesIO()
        Image.new("RGB", (16, 16), "red").save(stream, format="PNG")
        cls.image = base64.b64encode(stream.getvalue()).decode()

    def test_api_validation_and_inference_failure(self):
        from fastapi.testclient import TestClient
        from inference.server import create_app

        def predict(image, request):
            self.assertEqual(image.size, (16, 16))
            if request.goal == "fail":
                raise RuntimeError("test backend failure")
            return probabilities()

        payload = {"request_id": 1, "image_base64": self.image, "goal": "Approach the chair"}
        with TestClient(create_app(predictor=predict)) as client:
            self.assertTrue(client.get("/health").json()["ready"])
            response = client.post("/decide", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["probabilities"], probabilities())
            for change in ({"image_base64": "!!!!"}, {"goal": " "}, {"request_id": True},
                           {"previous_action": "backward"}, {"unexpected": 1}):
                self.assertEqual(client.post("/decide", json={**payload, **change}).status_code, 422)
            self.assertEqual(client.post("/decide", content=b"x" * 3_000_001).status_code, 413)
            self.assertEqual(client.post("/decide", json={**payload, "goal": "fail"}).status_code, 503)
            self.assertEqual(client.post("/decide", json=payload).status_code, 200)


if __name__ == "__main__":
    unittest.main()
