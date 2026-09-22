"""Shared action contract and ROS-independent command lifetime rules."""

import math
from dataclasses import dataclass

ACTIONS = ("forward", "left", "right", "stop", "goal_reached")
OPTIONS = (
    "Move forward a short distance.",
    "Rotate left in place a small amount.",
    "Rotate right in place a small amount.",
    "Stop because the view is unclear, the path is blocked, or no action is appropriate.",
    "Stop because the goal has already been reached.",
)
QUESTION = "Which short action should the robot take next toward the goal?"


def validate_probabilities(values):
    if not isinstance(values, dict) or set(values) != set(ACTIONS):
        raise ValueError("Response must contain exactly the five action probabilities")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
           for p in values.values()):
        raise ValueError("Probabilities must be finite numbers between zero and one")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-4):
        raise ValueError("Probabilities must sum to one")
    return values


def choose_action(values, min_probability, min_margin):
    values = validate_probabilities(values)
    ranked = sorted(ACTIONS, key=values.__getitem__, reverse=True)
    best, second = ranked[:2]
    if values[best] < min_probability or values[best] - values[second] < min_margin:
        return "stop"
    return best


@dataclass
class MotionState:
    """Generation IDs prevent a late response from restarting a stopped robot."""

    enabled: bool = False
    generation: int = 0
    action: str = "stop"
    deadline: float = 0.0

    def enable(self, enabled):
        self.generation += 1
        self.enabled = enabled
        self.stop()

    def stop(self):
        self.action = "stop"
        self.deadline = 0.0

    def accept(self, action, generation, deadline, now):
        if generation != self.generation or not self.enabled:
            return False
        if action not in ACTIONS or not math.isfinite(deadline) or now >= deadline:
            self.stop()
            return False
        if action == "goal_reached":
            self.enable(False)
        else:
            self.action, self.deadline = action, deadline
        return True

    def velocity(self, now, forward_speed, turn_speed, dry_run):
        if not self.enabled or now >= self.deadline:
            self.stop()
        if dry_run:
            return 0.0, 0.0
        return {
            "forward": (forward_speed, 0.0),
            "left": (0.0, turn_speed),
            "right": (0.0, -turn_speed),
        }.get(self.action, (0.0, 0.0))
