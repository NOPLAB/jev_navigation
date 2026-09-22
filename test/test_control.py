"""Run: python -m unittest discover -s test -v. No ROS or GPU required."""

import base64
import io
import unittest

from jev_navigation.policy import ACTIONS, MotionState, choose_action, validate_probabilities


def probabilities(action="forward"):
    return {key: float(key == action) for key in ACTIONS}


class ControlTest(unittest.TestCase):
    def test_command_lifetime_and_restart(self):
        state = MotionState()
        self.assertFalse(state.accept("forward", 0, 10.0, 1.0))
        state.enable(True)
        generation = state.generation
        self.assertTrue(state.accept("forward", generation, 2.0, 1.0))
        self.assertEqual(state.velocity(1.1, .1, .25, False), (.1, 0))
        self.assertEqual(state.velocity(1.2, .1, .25, True), (0, 0))
        self.assertEqual(state.deadline, 2.0)  # Publishing never renews the lease.
        self.assertEqual(state.velocity(2.0, .1, .25, False), (0, 0))
        state.enable(False)
        state.enable(True)
        self.assertFalse(state.accept("left", generation, 5.0, 2.1))
        self.assertEqual(state.velocity(2.2, .1, .25, False), (0, 0))
        self.assertFalse(state.accept("forward", state.generation, 2.0, 2.1))
        self.assertTrue(state.accept("right", state.generation, 3.0, 2.1))
        self.assertEqual(state.velocity(2.2, .1, .25, False), (0, -.25))
        self.assertTrue(state.accept("goal_reached", state.generation, 3.0, 2.2))
        self.assertFalse(state.enabled)
        self.assertEqual(state.velocity(2.3, .1, .25, False), (0, 0))

    def test_uncertain_and_invalid_outputs(self):
        self.assertEqual(choose_action(dict.fromkeys(ACTIONS, .2), .2, 0.0), "forward")
        self.assertEqual(choose_action(dict(forward=.26, left=.17, right=.21,
                                            stop=.28, goal_reached=.08), .2, 0.0), "stop")
        self.assertEqual(choose_action(probabilities(), .6, .15), "forward")
        self.assertEqual(choose_action(dict.fromkeys(ACTIONS, .2), .6, .15), "stop")
        values = dict.fromkeys(ACTIONS, 0.0)
        values.update(forward=.51, left=.49)
        self.assertEqual(choose_action(values, .5, .15), "stop")
        for bad in (float("nan"), float("inf"), -1, 2, True, "1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_probabilities({**probabilities(), "forward": bad})
        for values in ({}, dict.fromkeys(ACTIONS, .1), {**probabilities(), "extra": 0}):
            with self.assertRaises(ValueError):
                validate_probabilities(values)


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
