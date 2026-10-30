"""Offline contracts and failure tests, NOT native-physics crossing results."""

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from amphibian_dynamics import AmphibianForces
from crossing_control import ActuatorAllocator, CrossingController, CrossingMonitor, ReferenceSwitcher, WaypointTracker, route_preflight, crossing_checkpoint
from river_geometry import RiverGeometry

CFG = json.loads((PROJECT / "configs/crossing_experiment.json").read_text())
ROBOT = json.loads((PROJECT / "configs/robot_v1.json").read_text())
SCENE = json.loads((PROJECT / "configs/river_forest.json").read_text())
FIELD = RiverGeometry(SCENE)
WEIGHT = ROBOT["mass_kg"] * ROBOT["gravity_m_s2"]


def sample(t, x=-12., y=-23., afloat=False, speed=.4):
    row = {"time_s": float(t), "x_m": x, "y_m": y, "z_m": .03544 if afloat else FIELD.mesh_height(x, y) + .23,
           "yaw_rad": 0., "vx_m_s": speed, "vy_m_s": 0., "vz_m_s": 0., "wz_rad_s": 0.,
           "roll_deg": 0., "pitch_deg": 0., "left_normal_n": 0. if afloat else WEIGHT/2,
           "right_normal_n": 0. if afloat else WEIGHT/2, "buoyancy_n": WEIGHT if afloat else 0.,
           "left_thrust_n": 15. if afloat else 0., "right_thrust_n": 15. if afloat else 0.,
           "track_mechanical_power_w": 10., "thruster_ideal_power_proxy_w": 5.}
    return row


def controller():
    points = [[x, -23.] for x in np.linspace(-12, 12, 25)]
    return CrossingController(CFG, ROBOT, FIELD, [-12, -23], [12, -23], points)


class Water:
    default_ground_mu = .8
    def __init__(self, wet): self.wet = wet
    def water_columns(self, x, y):
        return np.zeros_like(x), np.full_like(x, -3.), np.full_like(x, float(self.wet))


class CrossingContracts(unittest.TestCase):
    def test_photos_require_entry_float_and_dry_opposite_bank_states(self):
        c = controller()
        s = FIELD.cross_section(-23.)
        entry = sample(8, x=s["center_x"]-s["shore_half_width"])
        entry["buoyancy_n"] = WEIGHT * .25
        c.phase = "enter_water"
        self.assertEqual(crossing_checkpoint(entry,c,ROBOT,FIELD,[]), "01_entering_water")
        water = sample(15,x=s["center_x"],afloat=True)
        c.phase = "swim"; c.monitor.afloat_s = 1.5; c.monitor.afloat_seen = True
        self.assertEqual(crossing_checkpoint(water,c,ROBOT,FIELD,["01_entering_water"]), "02_swimming")
        self.assertIsNone(crossing_checkpoint(water,c,ROBOT,FIELD,["01_entering_water","02_swimming"]))
        land = sample(25,x=s["center_x"]+s["shore_half_width"]+1.2)
        c.phase = "land_finish"
        self.assertEqual(crossing_checkpoint(land,c,ROBOT,FIELD,["01_entering_water","02_swimming"]), "03_landing")

    def test_course_clearance_and_floating_depth(self):
        result = route_preflight(CFG, SCENE, ROBOT, FIELD)
        self.assertTrue(result["passed"])
        self.assertGreater(result["minimum_obstacle_envelope_clearance_m"], .6)
        self.assertGreater(result["maximum_water_depth_m"], .5)

    def test_float_classification_uses_support_not_origin_height(self):
        switcher = ReferenceSwitcher(ROBOT, FIELD, CFG)
        x = FIELD.cross_section(-23)["center_x"]
        for i in range(45):
            mode = switcher.update(sample(i*.05, x=x, afloat=True), .05)
        self.assertEqual(mode, "WATER")
        self.assertGreater(sample(0, x=x, afloat=True)["z_m"], 0.)
        for i in range(45, 90):
            mode = switcher.update(sample(i*.05, x=12.), .05)
        self.assertEqual(mode, "LAND")

    def test_positive_yaw_produces_positive_native_force_model_torque(self):
        for mode in ["LAND", "WATER"]:
            row = sample(0, afloat=mode == "WATER", speed=0)
            allocation = ActuatorAllocator(ROBOT).commands([.2, .2], mode, row, .05)
            expected = allocation["tracks"] if mode == "LAND" else allocation["thrusters"]
            self.assertGreater(expected[1], expected[0])
            model = AmphibianForces(ROBOT)
            position = [0, 0, .23 if mode == "LAND" else model.equilibrium_z]
            contacts = [{"normal_n": WEIGHT/2 if mode == "LAND" else 0., "mu": .8, "normal": [0, 0, 1]}] * 2
            force, torque, state = model.compute(position, [0, 0, 0, 1], [0]*6, allocation,
                                                 contacts, Water(mode == "WATER"), .05)
            self.assertGreater(torque[2], 0.)
            self.assertGreater(force[0], 0.)

    def test_water_mode_cannot_command_tracks(self):
        command = ActuatorAllocator(ROBOT).commands([1, 1], "WATER", sample(0, afloat=True), .05)
        self.assertEqual(command["tracks"], [0., 0.])
        self.assertTrue(all(-1 <= x <= 1 for x in command["thrusters"]))

    def test_land_mode_cannot_command_thrusters(self):
        command = ActuatorAllocator(ROBOT).commands([1, -1], "LAND", sample(0), .05)
        self.assertEqual(command["thrusters"], [0., 0.])
        self.assertTrue(all(-1 <= x <= 1 for x in command["tracks"]))

    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            ActuatorAllocator(ROBOT).commands([math.nan, 0], "LAND", sample(0), .05)

    def test_lateral_error_steers_back_toward_course(self):
        tracker = WaypointTracker([[-12, -23], [12, -23]], CFG)
        action, _ = tracker.action(sample(0, y=-24), "LAND")
        self.assertGreater(action[1], 0.)
        action, _ = tracker.action(sample(0, y=-22), "LAND")
        self.assertLess(action[1], 0.)

    def test_stall_finishes_with_zero_commands(self):
        c = controller()
        for i in range(300):
            command = c.tick(sample(i*.05, speed=0))
            if c.reason: break
        self.assertEqual(c.reason, "no_forward_progress")
        self.assertEqual(command["tracks"] + command["thrusters"], [0.]*4)

    def test_attitude_failure_finishes_with_zero_commands(self):
        c = controller()
        row = sample(0); row["roll_deg"] = 40.
        command = c.tick(row)
        self.assertEqual(c.reason, "attitude_limit")
        self.assertEqual(command["tracks"] + command["thrusters"], [0.]*4)

    def test_delayed_observation_cannot_hold_old_command(self):
        c = controller()
        c.tick(sample(0))
        command = c.tick(sample(.3))
        self.assertEqual(c.reason, "control_timing_gap")
        self.assertEqual(command["tracks"] + command["thrusters"], [0.]*4)

    def test_goal_shortcut_without_water_cannot_pass(self):
        monitor = CrossingMonitor(CFG, ROBOT, FIELD, [-12, -23], [12, -23], 150)
        monitor.observe(sample(0, speed=0), "LAND")
        end = sample(1, x=12., speed=0)
        monitor.observe(end, "LAND")
        summary = monitor.summary(end, "goal_reached_and_stopped", "offline_test_only")
        self.assertFalse(summary["crossing_passed"])
        self.assertFalse(summary["checks"]["entered_water"])

    def test_synthetic_measurement_sequence_covers_full_lifecycle(self):
        # Scripted observations test sequencing and stop logic only. No dynamics are simulated here.
        c = controller()
        for i in range(1500):
            t = i * .05
            x = min(11.65, -12 + max(0., t-2.) * .4)
            afloat = -4.8 < x < 1.5
            speed = .4 if 2 < t < 61.125 else 0.
            row = sample(t, x=x, afloat=afloat, speed=speed)
            command = c.tick(row)
            if c.reason: break
        self.assertEqual(c.reason, "goal_reached_and_stopped")
        self.assertEqual(command["tracks"] + command["thrusters"], [0.]*4)
        self.assertTrue(c.monitor.summary(row, c.reason, "offline_test_only")["crossing_passed"])
        phases = {x["phase"] for x in c.decisions}
        self.assertTrue({"settle", "land_approach", "swim", "land_finish", "brake"}.issubset(phases))
        json.dumps(c.monitor.summary(row, c.reason, "offline_test_only"), allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
