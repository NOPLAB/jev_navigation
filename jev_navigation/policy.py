"""Shared action contract and ROS-independent command lifetime rules."""

import math
from dataclasses import dataclass

CURVATURES = {"forward": 0.0, "gentle_left": 1.0, "gentle_right": -1.0,
              "left": 2.0, "right": -2.0}
ACTIONS = (*CURVATURES, "stop", "goal_reached")
OPTIONS = (
    "Follow a short straight path forward toward the goal.",
    "Follow a short gentle left curve forward toward the goal.",
    "Follow a short gentle right curve forward toward the goal.",
    "Follow a short tighter left curve forward toward the goal.",
    "Follow a short tighter right curve forward toward the goal.",
    "Stop because immediate movement is obstructed or cannot be assessed.",
    "Finish because the goal is visibly satisfied; remain stationary.",
)
QUESTION = "Which short local path should the robot follow next toward the goal?"


def validate_probabilities(values):
    if not isinstance(values, dict) or set(values) != set(ACTIONS):
        raise ValueError("Response probabilities must match path candidates: " + ", ".join(ACTIONS))
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
           for p in values.values()):
        raise ValueError("Probabilities must be finite numbers between zero and one")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-4):
        raise ValueError("Probabilities must sum to one")
    return values


def choose_action(values, min_probability, min_margin, current="stop", switch_margin=0.0):
    values = validate_probabilities(values)
    ranked = sorted(ACTIONS, key=values.__getitem__, reverse=True)
    best, second = ranked[:2]
    if values[best] < min_probability or values[best] - values[second] < min_margin:
        return "stop"
    # Only stabilize moving paths. Never suppress a winning stop/goal result.
    if (best in CURVATURES and current in CURVATURES and values[current] >= min_probability
            and values[best] - values[current] < switch_margin):
        return current
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

    def active(self, now):
        if not self.enabled or now >= self.deadline:
            self.stop()
        return self.action in CURVATURES
