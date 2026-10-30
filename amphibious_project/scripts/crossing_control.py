"""Pure control/metrics for a native-physics river crossing; no Isaac imports.

Commands are belt speed (m/s) and normalized thrust, never robot pose/velocity.
The reference switcher is a diagnostic baseline, not CD-HSSRL's HSSP.
"""

from collections import deque
import math

import numpy as np


MODE_TO_OPTION = {"WATER": 0, "HYBRID": 1, "LAND": 2}
OPTION_TO_MODE = {v: k for k, v in MODE_TO_OPTION.items()}


def wrap_angle(value):
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


def load_ratio(row, robot):
    return (row["left_normal_n"] + row["right_normal_n"]) / (robot["mass_kg"] * robot["gravity_m_s2"])


def body_forward_speed(row):
    return row["vx_m_s"] * math.cos(row["yaw_rad"]) + row["vy_m_s"] * math.sin(row["yaw_rad"])


def crossing_checkpoint(row, controller, robot, field, captured):
    """Pick three measured physical states for photos, in crossing order."""
    section = field.cross_section(row["y_m"])
    x = row["x_m"]
    weight = robot["mass_kg"] * robot["gravity_m_s2"]
    buoy = row["buoyancy_n"] / weight
    support = load_ratio(row, robot)
    if not captured:
        if (controller.phase == "enter_water" and x < section["center_x"]
                and buoy > .10 and support > .10):
            return "01_entering_water"
    elif "02_swimming" not in captured:
        if (controller.phase == "swim" and controller.monitor.afloat_s >= 1.
                and support < .08 and buoy > .80 and x >= section["center_x"] - .4):
            return "02_swimming"
    elif "03_landing" not in captured:
        if (controller.monitor.afloat_seen and support > .5 and buoy < .12
                and x > section["center_x"] + section["shore_half_width"] + .5
                and field.mesh_height(x, row["y_m"]) > field.water_z + .12):
            return "03_landing"
    return None


class ReferenceSwitcher:
    """Contact, buoyancy and local depth with low-pass filtering and dwell time."""

    def __init__(self, robot, field, config):
        self.robot, self.field, self.config = robot, field, config
        self.mode = "LAND"
        self.filtered_load = 1.0
        self.candidate = self.mode
        self.candidate_since = 0.0

    def update(self, row, dt):
        self.filtered_load += (1 - math.exp(-dt / .25)) * (load_ratio(row, self.robot) - self.filtered_load)
        water_depth = max(0., self.field.water_z - self.field.mesh_height(row["x_m"], row["y_m"]))
        buoy_ratio = row["buoyancy_n"] / (self.robot["mass_kg"] * self.robot["gravity_m_s2"])
        # Land is not inferred from body-origin z: a floating vehicle's origin is ABOVE water.
        if self.filtered_load > .38 and buoy_ratio < .06 and water_depth < .015:
            candidate = "LAND"
        elif (self.filtered_load < .08 and buoy_ratio > .72
              and water_depth > self.config["shore_transition_depth_m"]):
            candidate = "WATER"
        else:
            candidate = "HYBRID"
        now = row["time_s"]
        if candidate != self.candidate:
            self.candidate, self.candidate_since = candidate, now
        if now - self.candidate_since >= (.4 if candidate != "HYBRID" else .10):
            self.mode = candidate
        return self.mode


class WaypointTracker:
    def __init__(self, points, config):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 2:
            raise ValueError("At least two finite XY waypoints are required")
        if not np.isfinite(self.points).all():
            raise ValueError("Non-finite waypoint")
        self.index = 1
        self.config = config

    def target(self, row):
        xy = np.array([row["x_m"], row["y_m"]])
        while self.index < len(self.points) - 1:
            delta = self.points[self.index] - xy
            segment = self.points[self.index] - self.points[self.index - 1]
            passed = float(delta @ segment) <= 0
            if np.linalg.norm(delta) < .65 or passed:
                self.index += 1
            else:
                break
        return self.points[self.index]

    def action(self, row, mode):
        target = self.target(row)
        delta = target - [row["x_m"], row["y_m"]]
        error = wrap_angle(math.atan2(delta[1], delta[0]) - row["yaw_rad"])
        yaw_rate = float(np.clip(1.6 * error - .35 * row["wz_rad_s"],
                                 -self.config["max_yaw_rate_rad_s"], self.config["max_yaw_rate_rad_s"]))
        speed = self.config[{"LAND": "land_speed_m_s", "HYBRID": "hybrid_speed_m_s", "WATER": "water_speed_m_s"}[mode]]
        speed *= max(0., math.cos(error)) ** 2
        goal_distance = float(np.linalg.norm(self.points[-1] - [row["x_m"], row["y_m"]]))
        speed = min(speed, max(.12, .55 * goal_distance))
        # Internal normalized velocity reference, then allocated to four actuators.
        return np.array([speed / 1.5, yaw_rate / 2.0], dtype=np.float32), target


class PolylineTracker:
    """Pure-pursuit follower for one short observed-waypoint polyline.

    The OmniNav head returns five positions along one short arc (first point
    about 0.25 m, last about 0.8-1.0 m). Following only the first point forces
    an accelerate-stop cycle every inference; this tracker projects the robot
    onto the polyline and steers to a configurable lookahead point so the whole
    arc can be executed continuously.
    """

    def __init__(self, points, config):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("At least two finite XY polyline points are required")
        if not np.isfinite(pts).all():
            raise ValueError("Non-finite polyline point")
        keep = [0]
        for index in range(1, len(pts)):
            if float(np.linalg.norm(pts[index] - pts[keep[-1]])) > 1e-9:
                keep.append(index)
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("Polyline has no measurable length")
        self.config = config
        self.points = pts
        self.segments = np.diff(pts, axis=0)
        self.lengths = np.linalg.norm(self.segments, axis=1)
        self.cumulative = np.concatenate(([0.], np.cumsum(self.lengths)))
        self.length = float(self.cumulative[-1])
        self.lookahead = float(config.get("polyline_lookahead_m", .35))
        if not .05 <= self.lookahead <= 1.:
            raise ValueError("Invalid polyline lookahead")
        self.index = 0

    def project(self, row):
        """Return (travelled arc length, lateral distance) on the polyline."""
        xy = np.array([row["x_m"], row["y_m"]], dtype=float)
        best_distance, best_travelled = float("inf"), 0.
        for index in range(max(0, self.index - 1), len(self.points) - 1):
            start = self.points[index]
            vector = self.segments[index]
            length = float(self.lengths[index])
            along = float(np.clip((xy - start) @ vector / (length * length), 0., 1.))
            distance = float(np.linalg.norm(xy - (start + along * vector)))
            if distance < best_distance:
                best_distance = distance
                best_travelled = float(self.cumulative[index] + along * length)
                self.index = index
        return best_travelled, best_distance

    def point_at(self, travelled):
        travelled = float(np.clip(travelled, 0., self.length))
        index = int(np.searchsorted(self.cumulative, travelled, side="right") - 1)
        index = min(max(index, 0), len(self.lengths) - 1)
        along = (travelled - self.cumulative[index]) / self.lengths[index]
        return self.points[index] + along * self.segments[index]

    def remaining(self, row):
        return self.length - self.project(row)[0]

    def action(self, row, mode):
        xy = np.array([row["x_m"], row["y_m"]], dtype=float)
        travelled, _ = self.project(row)
        remaining = self.length - travelled
        target = self.point_at(min(self.length, travelled + self.lookahead))
        delta = target - xy
        error = wrap_angle(math.atan2(delta[1], delta[0]) - row["yaw_rad"])
        yaw_rate = float(np.clip(1.6 * error - .35 * row["wz_rad_s"],
                                 -self.config["max_yaw_rate_rad_s"], self.config["max_yaw_rate_rad_s"]))
        speed = self.config[{"LAND": "land_speed_m_s", "HYBRID": "hybrid_speed_m_s",
                             "WATER": "water_speed_m_s"}[mode]]
        speed *= max(0., math.cos(error)) ** 2
        speed = min(speed, max(.12, .9 * remaining))
        return np.array([speed / 1.5, yaw_rate / 2.], dtype=np.float32), target


class ActuatorAllocator:
    """Physical speed/yaw servos -> two tracks and two thrusters.

    For +yaw, right track/propeller must be stronger than the left.
    Uses bounded PI correction on land and linear/quadratic drag feedforward in water.
    """

    def __init__(self, robot):
        self.robot = robot
        self.integral = 0.
        self.last_mode = None

    def commands(self, normalized_action, mode, row, dt):
        action = np.asarray(normalized_action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected finite action [v_norm, yaw_rate_norm]")
        if mode not in MODE_TO_OPTION:
            raise ValueError("Unknown actuator mode")
        action = np.clip(action, -1, 1)
        v_ref, w_ref = action * [1.5, 2.0]
        v_ref = float(np.clip(v_ref, -.8, .8))
        w_ref = float(np.clip(w_ref, -.65, .65))
        if mode != self.last_mode:
            self.integral = 0.
            self.last_mode = mode
        error = v_ref - body_forward_speed(row)
        self.integral = float(np.clip(self.integral + error * dt, -.6, .6))
        tr = self.robot["tracks"]
        belt = v_ref + .9 * error + .8 * self.integral
        differential = .5 * tr["center_separation_m"] * (w_ref + .7 * (w_ref - row["wz_rad_s"]))
        tracks = np.clip([belt - differential, belt + differential], -tr["max_speed_m_s"], tr["max_speed_m_s"])
        hydro = self.robot["hydrodynamics"]
        thrust_x = hydro["linear_drag_body"][0] * v_ref + hydro["quadratic_drag_body"][0] * abs(v_ref) * v_ref
        thrust_x += 42. * error
        yaw_torque = (hydro["linear_angular_drag_body"][2] * w_ref
                      + hydro["quadratic_angular_drag_body"][2] * abs(w_ref) * w_ref
                      + 8. * (w_ref - row["wz_rad_s"]))
        th = self.robot["thrusters"]
        lever = abs(th["positions_body_m"][0][1])
        thrusts = np.array([thrust_x / 2 - yaw_torque / (2 * lever), thrust_x / 2 + yaw_torque / (2 * lever)])
        throttles = np.clip(thrusts / th["max_thrust_each_n"], -1, 1)
        # Avoid PI pushing against the water when the tracks have no support.
        if mode == "WATER":
            tracks[:] = 0
            self.integral = 0
        if mode == "LAND":
            throttles[:] = 0
        return {"mode": mode, "tracks": tracks.tolist(), "thrusters": throttles.tolist()}


class CrossingMonitor:
    def __init__(self, config, robot, field, start, goal, timeout):
        self.cfg, self.robot, self.field = config, robot, field
        self.start, self.goal = np.array(start), np.array(goal)
        self.timeout = timeout
        self.first_time = None
        self.previous = None
        self.path_length = 0.
        self.progress_window = deque()
        self.afloat_s = 0.
        self.was_afloat = False
        self.afloat_seen = False
        self.land_before = False
        self.land_after = False
        self.buoyancy_seen = False
        self.water_thrust_seen = False
        self.mode_switches = 0
        self.last_mode = None
        self.events = []
        self.max_roll = self.max_pitch = self.max_speed = 0.
        self.max_lateral = 0.
        self.energy_proxy_j = 0.

    def observe(self, row, mode):
        now = row["time_s"]
        if self.first_time is None:
            self.first_time = now
        dt = max(0., now - self.previous["time_s"]) if self.previous else 0.
        position = np.array([row["x_m"], row["y_m"]])
        if self.previous:
            self.path_length += math.hypot(row["x_m"] - self.previous["x_m"], row["y_m"] - self.previous["y_m"])
        line = self.goal - self.start
        unit = line / max(1e-9, np.linalg.norm(line))
        progress = float((position - self.start) @ unit)
        lateral = abs(float(np.cross(np.r_[unit, 0.], np.r_[position - self.start, 0.])[2]))
        support = load_ratio(row, self.robot)
        buoy = row["buoyancy_n"] / (self.robot["mass_kg"] * self.robot["gravity_m_s2"])
        if support > .4 and buoy < .12 and not self.afloat_seen:
            self.land_before = True
        if buoy > .15:
            self.buoyancy_seen = True
        afloat = support < self.cfg["ground_support_ratio_afloat"] and buoy > self.cfg["buoyancy_support_ratio_afloat"]
        if afloat:
            self.afloat_s += dt if self.was_afloat else 0.
            if not self.afloat_seen:
                self.events.append({"event": "first_free_float", "time_s": now, "xy_m": position.tolist()})
            self.afloat_seen = True
        self.was_afloat = afloat
        if self.afloat_seen and support > .4 and buoy < .12 and progress > np.linalg.norm(line) * .55:
            if not self.land_after:
                self.events.append({"event": "opposite_bank_contact", "time_s": now, "xy_m": position.tolist()})
            self.land_after = True
        if afloat and abs(row["left_thrust_n"]) + abs(row["right_thrust_n"]) > 2.:
            self.water_thrust_seen = True
        if mode != self.last_mode:
            if self.last_mode is not None:
                self.mode_switches += 1
            self.events.append({"event": "mode", "mode": mode, "time_s": now, "xy_m": position.tolist()})
            self.last_mode = mode
        self.max_roll = max(self.max_roll, abs(row["roll_deg"]))
        self.max_pitch = max(self.max_pitch, abs(row["pitch_deg"]))
        self.max_speed = max(self.max_speed, math.hypot(row["vx_m_s"], row["vy_m_s"]))
        self.max_lateral = max(self.max_lateral, lateral)
        self.energy_proxy_j += (row["track_mechanical_power_w"] + row["thruster_ideal_power_proxy_w"]) * dt
        self.progress_window.append((now, progress))
        while len(self.progress_window) > 1 and now - self.progress_window[1][0] > self.cfg["stall_window_s"]:
            self.progress_window.popleft()
        self.previous = dict(row)

    def failure(self, row, settling=False, braking=False):
        if not all(math.isfinite(float(v)) for k, v in row.items() if isinstance(v, (float, int))):
            return "non_finite_physics"
        if abs(row["roll_deg"]) > self.cfg["max_roll_deg"] or abs(row["pitch_deg"]) > self.cfg["max_pitch_deg"]:
            return "attitude_limit"
        if math.hypot(row["vx_m_s"], row["vy_m_s"]) > self.cfg["max_speed_m_s"]:
            return "speed_limit"
        if abs(row["x_m"]) > self.field.half_x - .5 or abs(row["y_m"]) > self.field.half_y - .5:
            return "left_terrain_bounds"
        if row["z_m"] < self.field.water_z - .55:
            return "excessive_submergence"
        if row["time_s"] - self.first_time > self.timeout:
            return "simulation_timeout"
        if not settling and not braking and self.progress_window:
            a, b = self.progress_window[0], self.progress_window[-1]
            if b[0] - a[0] >= self.cfg["stall_window_s"] - .06 and b[1] - a[1] < self.cfg["stall_min_progress_m"]:
                return "no_forward_progress"
        return None

    def summary(self, row, reason, kind):
        distance = math.hypot(row["x_m"] - self.goal[0], row["y_m"] - self.goal[1]) if row else None
        stopped = bool(row and math.hypot(row["vx_m_s"], row["vy_m_s"]) < self.cfg["goal_speed_m_s"])
        checks = {"reached_goal": bool(distance is not None and distance <= self.cfg["goal_radius_m"]),
                  "stopped_at_goal": stopped, "started_on_land": self.land_before,
                  "entered_water": self.buoyancy_seen,
                  "free_float_over_minimum": self.afloat_s >= self.cfg["minimum_afloat_s"],
                  "effective_water_thrust": self.water_thrust_seen, "reached_opposite_dry_bank": self.land_after,
                  "terminated_normally": reason == "goal_reached_and_stopped"}
        return {"kind": kind, "result": reason, "crossing_passed": bool(all(checks.values())), "checks": checks,
                "elapsed_sim_s": row["time_s"] - self.first_time if row else 0., "path_length_m": self.path_length,
                "final_goal_distance_m": distance, "free_float_duration_s": self.afloat_s,
                "mode_switch_count": self.mode_switches, "max_abs_roll_deg": self.max_roll,
                "max_abs_pitch_deg": self.max_pitch, "max_horizontal_speed_m_s": self.max_speed,
                "max_distance_from_start_goal_line_m": self.max_lateral,
                "mechanical_and_ideal_propulsion_energy_proxy_j": self.energy_proxy_j,
                "energy_is_battery_measurement": False, "events": self.events}


class CrossingController:
    """Full land -> entry -> swim -> exit -> goal/brake lifecycle at 20 Hz."""

    def __init__(self, config, robot, field, start, goal, points):
        self.cfg, self.robot, self.field = config, robot, field
        self.start, self.goal = np.array(start), np.array(goal)
        self.switcher = ReferenceSwitcher(robot, field, config)
        self.tracker = WaypointTracker(points, config)
        self.allocator = ActuatorAllocator(robot)
        self.monitor = CrossingMonitor(config, robot, field, start, goal, config["crossing_timeout_s"])
        self.reason = None
        self.phase = "settle"
        self.last_time = None
        self.brake_since = self.hold_since = None
        self.decisions = []

    def zero(self, mode=None):
        return {"mode": mode or self.switcher.mode, "tracks": [0., 0.], "thrusters": [0., 0.]}

    def finish(self, reason):
        self.reason = reason
        return self.zero()

    def tick(self, row):
        if self.reason:
            return self.zero()
        now = row["time_s"]
        dt = now - self.last_time if self.last_time is not None else 1. / self.cfg["control_hz"]
        if dt <= 0 or dt > .2:
            return self.finish("control_timing_gap")
        self.last_time = now
        mode = self.switcher.update(row, dt)
        self.monitor.observe(row, mode)
        elapsed = now - self.monitor.first_time
        settling = elapsed < self.cfg["settle_s"]
        distance = float(np.linalg.norm(self.goal - [row["x_m"], row["y_m"]]))
        failure = self.monitor.failure(row, settling, self.brake_since is not None)
        if failure:
            return self.finish(failure)
        action = np.zeros(2)
        target = self.goal.copy()
        if settling:
            self.phase, command = "settle", self.zero(mode)
        else:
            if distance <= self.cfg["goal_radius_m"] and load_ratio(row, self.robot) > .35:
                if self.brake_since is None:
                    self.brake_since = now
            if self.brake_since is not None:
                self.phase, command = "brake", self.zero(mode)
                speed = math.hypot(row["vx_m_s"], row["vy_m_s"])
                if distance <= self.cfg["goal_radius_m"] and speed < self.cfg["goal_speed_m_s"]:
                    if self.hold_since is None:
                        self.hold_since = now
                    if now - self.hold_since >= self.cfg["goal_hold_s"]:
                        self.finish("goal_reached_and_stopped")
                else:
                    self.hold_since = None
                if now - self.brake_since > self.cfg["goal_brake_timeout_s"]:
                    self.finish("goal_stop_failed")
            else:
                if mode == "WATER":
                    self.phase = "swim"
                elif mode == "HYBRID":
                    self.phase = "exit_water" if self.monitor.afloat_seen else "enter_water"
                else:
                    self.phase = "land_finish" if self.monitor.afloat_seen else "land_approach"
                action, target = self.tracker.action(row, mode)
                command = self.allocator.commands(action, mode, row, dt)
        if self.reason:
            command = self.zero(mode)
        self.decisions.append({"observation_time_s": now, "phase": self.phase, "selected_mode": mode,
                               "target_x_m": float(target[0]), "target_y_m": float(target[1]),
                               "goal_distance_m": distance, "v_reference_m_s": float(action[0] * 1.5),
                               "yaw_rate_reference_rad_s": float(action[1] * 2.),
                               "filtered_support_ratio": self.switcher.filtered_load,
                               "next_left_track_m_s": command["tracks"][0], "next_right_track_m_s": command["tracks"][1],
                               "next_left_thruster": command["thrusters"][0], "next_right_thruster": command["thrusters"][1],
                               "control_dt_s": dt, "terminal_reason": self.reason or ""})
        return command


def route_preflight(config, scene_config, robot, field):
    """Check the chosen course without rendering or accessing a server."""
    mission = next(m for m in scene_config["missions"] if m["id"] == config["mission"])
    start, goal = np.array(mission["start_xy_m"]), np.array(mission["goal_xy_m"])
    length = float(np.linalg.norm(goal - start))
    count = int(math.ceil(length / config["waypoint_spacing_m"]))
    points = np.linspace(start, goal, count + 1)
    trees, rocks = field.objects()
    robot_radius = config["robot_clearance_radius_m"]
    obstacles = [(t["x"], t["y"], t["radius"]) for t in trees]
    obstacles += [(r["x"], r["y"], max(r["radii"][:2])) for r in rocks]
    profile, min_clearance = [], float("inf")
    samples = np.linspace(start, goal, int(length / .1) + 1)
    for xy in samples:
        x, y = map(float, xy)
        h = field.mesh_height(x, y)
        dx = (field.mesh_height(x + .1, y) - field.mesh_height(x - .1, y)) / .2
        dy = (field.mesh_height(x, y + .1) - field.mesh_height(x, y - .1)) / .2
        mu = scene_config["materials"][field.soil(x, y)]["dynamic_friction"]
        clearance = min(math.hypot(x - ox, y - oy) - radius - robot_radius for ox, oy, radius in obstacles)
        min_clearance = min(min_clearance, clearance)
        profile.append({"x_m": x, "y_m": y, "terrain_z_m": h, "water_depth_m": max(0., field.water_z - h),
                        "slope_deg": math.degrees(math.atan(math.hypot(dx, dy))),
                        "soil": field.soil(x, y), "dynamic_friction": mu,
                        "obstacle_envelope_clearance_m": clearance})
    dry_slopes = [p["slope_deg"] for p in profile if p["water_depth_m"] == 0]
    checks = {"course_inside_terrain": bool(all(abs(x) < field.half_x - robot_radius and abs(y) < field.half_y - robot_radius for x, y in samples)),
              "no_obstacle_envelope_intersection": min_clearance > 0,
              "start_and_goal_dry": field.mesh_height(*start) > field.water_z + .1 and field.mesh_height(*goal) > field.water_z + .1,
              "has_depth_for_free_float": max(p["water_depth_m"] for p in profile) > .40,
              "dry_slope_below_pitch_limit": max(dry_slopes) < config["max_pitch_deg"] - 5,
              "control_divides_physics_rate": config["physics_hz"] % config["control_hz"] == 0,
              "matches_robot_physics_rate": robot["physics_hz"] == config["physics_hz"]}
    return {"passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()},
            "mission": config["mission"], "start_xy_m": start.tolist(), "goal_xy_m": goal.tolist(),
            "route_length_m": length, "minimum_obstacle_envelope_clearance_m": min_clearance,
            "maximum_dry_slope_deg": max(dry_slopes), "maximum_water_depth_m": max(p["water_depth_m"] for p in profile),
            "waypoints_xy_m": points.tolist(), "profile": profile,
            "is_physics_crossing_validation": False,
            "limitation": "Analytic scene-config check; actual contacts, bank climbing and closed-loop crossing require Isaac Sim."}
