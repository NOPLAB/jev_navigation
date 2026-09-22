"""Short odom-frame arcs and pure-pursuit tracking; no ROS dependencies."""

import math

from .policy import CURVATURES


def make_path(action, pose, length):
    x, y, yaw = pose
    curvature = CURVATURES[action]
    points = []
    for i in range(31):
        s = length * i / 30
        dx = math.sin(curvature * s) / curvature if curvature else s
        dy = (1 - math.cos(curvature * s)) / curvature if curvature else 0.0
        points.append((x + math.cos(yaw) * dx - math.sin(yaw) * dy,
                       y + math.sin(yaw) * dx + math.cos(yaw) * dy))
    return points


class PathFollower:
    def __init__(self):
        self.points = []
        self.index = 0
        self.linear = self.angular = 0.0

    def reset(self):
        self.points = []
        self.index = 0
        self.linear = self.angular = 0.0

    def set_path(self, points):
        self.points = points
        self.index = 0  # Preserve output velocities across normal replanning.

    def target(self, pose, lookahead, tolerance, max_error, speed, max_turn, deceleration):
        if not self.points:
            return None
        x, y, yaw = pose
        distance = lambda point: math.hypot(point[0] - x, point[1] - y)
        self.index = min(range(self.index, len(self.points)), key=lambda i: distance(self.points[i]))
        if distance(self.points[self.index]) > max_error:
            return None
        remaining = sum(math.dist(a, b) for a, b in
                        zip(self.points[self.index:], self.points[self.index + 1:]))
        if distance(self.points[-1]) <= tolerance or remaining <= tolerance:
            return None
        target = self.points[-1]
        for point in self.points[self.index:]:
            if distance(point) >= lookahead:
                target = point
                break
        dx, dy = target[0] - x, target[1] - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if local_x <= 0 or dx * dx + dy * dy < 1e-8:
            return None
        curvature = 2 * local_y / (dx * dx + dy * dy)
        linear = min(speed, math.sqrt(2 * deceleration * max(0.0, remaining - tolerance)))
        if curvature:
            linear = min(linear, max_turn / abs(curvature))
        return linear, linear * curvature

    def slew(self, target, dt, acceleration, angular_acceleration):
        # Bound the ramp after a delayed control tick; safety stops bypass this.
        dt = max(0.0, min(dt, 0.1))
        def approach(current, wanted, limit):
            return current + max(-limit * dt, min(limit * dt, wanted - current))
        self.linear = approach(self.linear, target[0], acceleration)
        self.angular = approach(self.angular, target[1], angular_acceleration)
        return self.linear, self.angular
