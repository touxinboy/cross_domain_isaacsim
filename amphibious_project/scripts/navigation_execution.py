"""Execution adapter: one 5-point OmniNav polyline -> four physical commands.

No Isaac/Habitat imports and no pose/velocity writes. This is an execution
adapter, not an obstacle planner or a learned amphibious switching policy.

The dedicated head predicts five positions along one short arc (first point
about 0.25 m, last about 0.8-1.0 m) plus five separately predicted headings.
The previous adapter executed only the first point and then zeroed the command,
which forced an accelerate-stop cycle per inference and discarded the predicted
turn whenever the first point fell inside the reach tolerance. This adapter
follows the whole polyline with pure pursuit and treats a degenerate polyline
(total length below the configured threshold) as an in-place turn.
"""
import math

import numpy as np

from crossing_control import (ActuatorAllocator, PolylineTracker, ReferenceSwitcher,
                              wrap_angle)
from navigation_contracts import finite_vector


def safe_polyline_prefix(path, segment_is_safe):
    """Longest prefix whose every segment passes the observed-space check."""
    points = [[float(point[0]), float(point[1])] for point in path]
    prefix = [points[0]]
    for start, end in zip(points, points[1:]):
        if not segment_is_safe(start, end):
            break
        prefix.append(end)
    length = sum(math.dist(a, b) for a, b in zip(prefix, prefix[1:]))
    return prefix, length


class PhysicalNavigationBridge:
    def __init__(self, robot, field, control_config, max_plan_age_s=2., max_step_m=1.5):
        if not math.isfinite(max_plan_age_s) or not math.isfinite(max_step_m) or min(max_plan_age_s, max_step_m) <= 0:
            raise ValueError("Plan age and step limit must be positive finite numbers")
        self.switcher = ReferenceSwitcher(robot, field, control_config)
        self.allocator = ActuatorAllocator(robot)
        self.config = control_config
        self.reach_tolerance = float(control_config.get('short_waypoint_tolerance_m', .15))
        if not math.isfinite(self.reach_tolerance) or not .02 <= self.reach_tolerance <= .25:
            raise ValueError('Invalid short waypoint tolerance')
        self.finish_tolerance = float(control_config.get('polyline_finish_tolerance_m', .10))
        if not math.isfinite(self.finish_tolerance) or not .03 <= self.finish_tolerance <= .30:
            raise ValueError('Invalid polyline finish tolerance')
        self.degenerate_length = float(control_config.get('polyline_degenerate_length_m', .08))
        if not math.isfinite(self.degenerate_length) or not .01 <= self.degenerate_length <= .25:
            raise ValueError('Invalid polyline degenerate length')
        self.max_segment = float(control_config.get('polyline_max_segment_m', .60))
        if not math.isfinite(self.max_segment) or not .10 <= self.max_segment <= 1.0:
            raise ValueError('Invalid polyline segment limit')
        self.max_path = min(max_step_m, float(control_config.get('polyline_max_length_m', 1.5)))
        self.turn_tolerance = .06
        self.yaw_rate_tolerance = .08
        self.max_age = max_plan_age_s
        self.max_step = max_step_m
        self.plan = self.tracker = None
        self.latest_capture_time = -1.

    def clear_plan(self):
        self.plan = self.tracker = None
        self.allocator.integral = 0.

    def _validate_path(self, path):
        points = [[float(point[0]), float(point[1])] for point in path]
        if len(points) < 2 or not np.isfinite(np.asarray(points)).all():
            raise ValueError("A finite polyline with at least two points is required")
        lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
        if any(length > self.max_segment + 1e-9 for length in lengths):
            raise ValueError("Head predicted a polyline segment beyond the configured limit")
        total = sum(lengths)
        if total > self.max_path + 1e-9:
            raise ValueError("Head predicted a polyline beyond the configured physical limit")
        return points, lengths, total

    def accept_head(self, result, now_s, current_pose=None):
        """Only the dedicated head's output is accepted, never generic chat text."""
        if result.get("source") != "OmniNav_Slowfast_dedicated_head":
            raise ValueError("A dedicated OmniNav waypoint-head result is required")
        captured = float(result["capture_time_s"])
        now = float(now_s)
        if (not math.isfinite(captured) or not math.isfinite(now) or captured < 0
                or captured <= self.latest_capture_time or not 0 <= now - captured <= self.max_age):
            raise ValueError("The waypoint result is stale, future-dated or out of order")
        anchor = finite_vector(result["anchor_pose_xyyaw"], 3, "capture anchor")
        if current_pose is not None:
            pose = finite_vector(current_pose, 3, "current self pose")
            yaw_delta = abs((pose[2] - anchor[2] + math.pi) % (2 * math.pi) - math.pi)
            if math.hypot(pose[0] - anchor[0], pose[1] - anchor[1]) > .35 or yaw_delta > math.radians(20):
                raise ValueError("Robot moved too far from the head's capture anchor")
        turns = finite_vector(result["turn_angles_rad"], 5, "head angles")
        waypoints = result.get("waypoints_normalized")
        if not isinstance(waypoints, list) or len(waypoints) != 5:
            raise ValueError("Expected five waypoints from the dedicated OmniNav head")
        points = []
        for waypoint, turn in zip(waypoints, turns):
            offset = finite_vector(waypoint, 2, "head waypoint")
            radius = math.hypot(offset[0], offset[1]) * .3
            point = [anchor[0] + radius * math.cos(anchor[2] + turn),
                     anchor[1] + radius * math.sin(anchor[2] + turn)]
            if math.hypot(point[0] - anchor[0], point[1] - anchor[1]) > self.max_step:
                raise ValueError("Head predicted a point beyond the configured physical limit")
            points.append(point)
        path, lengths, total = self._validate_path([[anchor[0], anchor[1]]] + points)
        first_yaw = wrap_angle(anchor[2] + turns[0])
        if total <= 1e-9:
            goal_yaw = first_yaw
            self.tracker = None
        else:
            goal_yaw = wrap_angle(anchor[2] + turns[-1])
            self.tracker = PolylineTracker(path, self.config)
        self.plan = {
            "capture_time_s": captured,
            "anchor_pose_xyyaw": anchor,
            "path_xy_m": path,
            "path_length_m": total,
            "goal_xy_m": path[1],
            "final_goal_xy_m": path[-1],
            "first_yaw_rad": first_yaw,
            "goal_yaw_rad": goal_yaw,
        }
        self.latest_capture_time = captured
        return self.plan.copy()

    def truncate_plan(self, path):
        """Keep the longest observed-safe prefix and rebuild the follower."""
        if self.plan is None:
            raise ValueError("No active plan to truncate")
        points, lengths, total = self._validate_path(path)
        if math.dist(points[0], self.plan["path_xy_m"][0]) > 1e-6:
            raise ValueError("Truncated polyline must start at the capture anchor")
        if total <= 1e-9:
            raise ValueError("Truncated polyline has no length")
        goal_yaw = wrap_angle(math.atan2(points[-1][1] - points[-2][1],
                                         points[-1][0] - points[-2][0]))
        plan = dict(self.plan)
        plan["path_xy_m"] = points
        plan["path_length_m"] = total
        plan["final_goal_xy_m"] = points[-1]
        plan["goal_yaw_rad"] = goal_yaw
        self.plan = plan
        self.tracker = PolylineTracker(points, self.config)
        return plan.copy()

    def _heading_action(self, row, goal_yaw):
        error = wrap_angle(goal_yaw - row["yaw_rad"])
        if abs(error) > self.turn_tolerance or abs(row["wz_rad_s"]) > self.yaw_rate_tolerance:
            yaw_rate = float(np.clip(1.6 * error - .35 * row["wz_rad_s"],
                                     -self.config["max_yaw_rate_rad_s"], self.config["max_yaw_rate_rad_s"]))
            return np.array([0., yaw_rate / 2.], dtype=np.float32), True
        return np.zeros(2, dtype=np.float32), False

    def tick(self, row, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Control dt must be positive and finite")
        for key in ["time_s", "x_m", "y_m", "yaw_rad", "vx_m_s", "vy_m_s", "wz_rad_s",
                    "buoyancy_n", "left_normal_n", "right_normal_n"]:
            if not math.isfinite(float(row[key])):
                raise ValueError("Non-finite measured control state: " + key)
        mode = self.switcher.update(row, dt)
        status, target = "no_plan", None
        action = np.zeros(2, dtype=np.float32)
        age = float(row["time_s"]) - self.plan["capture_time_s"] if self.plan is not None else None
        if age is not None:
            if age < 0 or age > self.max_age:
                status = "expired_plan"
            elif self.plan["path_length_m"] <= self.degenerate_length:
                target = self.plan["final_goal_xy_m"]
                action, turning = self._heading_action(row, self.plan["first_yaw_rad"])
                status = "heading_alignment" if turning else "short_waypoint_reached"
            else:
                target = self.plan["final_goal_xy_m"]
                remaining = self.tracker.remaining(row)
                if remaining <= self.finish_tolerance:
                    action, turning = self._heading_action(row, self.plan["goal_yaw_rad"])
                    status = "heading_alignment" if turning else "polyline_reached"
                else:
                    action, target = self.tracker.action(row, mode)
                    status = "tracking"
        if status in ["tracking", "heading_alignment"]:
            command = self.allocator.commands(action, mode, row, dt)
        else:
            # Zero actuator demand does not instantly halt a floating rigid body.
            self.allocator.integral = 0.
            command = {"mode": mode, "tracks": [0., 0.], "thrusters": [0., 0.]}
        return {"status": status, "plan_age_s": age,
                "target_xy_m": None if target is None else [float(target[0]), float(target[1])],
                "path_length_m": self.plan["path_length_m"] if self.plan is not None else None,
                "switch_policy": getattr(self.switcher, 'policy_name', 'reference_ground_truth_baseline'),
                "command": command}
