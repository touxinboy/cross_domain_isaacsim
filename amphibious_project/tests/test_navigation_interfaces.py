"""Exercise original MSGNav decisions and physical command/frame boundaries.

Uses deliberately synthetic observations and scripted responses. These tests
do not run Qwen/OmniNav inference or establish navigation performance.
"""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from msgnav_bridge import MSGNavDecision
from navigation_contracts import native_head_first_step, omni_local_point, pointgoal_positions
from navigation_execution import PhysicalNavigationBridge, safe_polyline_prefix
from omninav_pointgoal import prepare_request


class RecordedReplies:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.last_usage = None

    def msgnav_completion(self, system, contents):
        self.requests.append((system, contents))
        if not self.responses:
            raise AssertionError("Unexpected additional model request")
        return self.responses.pop(0)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="l17_msgnav_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        Image.new("RGB", (32, 32), (15, 100, 200)).save(self.root / "observed.png")
        self.obs = {"source": "synthetic_fixture", "instruction": "Find the orange jacket.",
                    "capture_time_s": 10., "anchor_pose_xyyaw": [0., 0., 0.],
                    "objects": [{"id": 1, "class_name": "jacket", "position_world_m": [4., 2., 1.], "approach_xy_m": [3., 2.]},
                                {"id": 2, "class_name": "tree", "position_world_m": [4., 3., 2.]}],
                    "snapshots": {"frame0": {"image": "observed.png", "approach_xy_m": [1., 2.]}},
                    "edges": [{"object_ids": [1, 2], "image_ids": ["frame0"]}],
                    "frontiers": [{"image": "observed.png", "approach_xy_m": [1., 0.]}],
                    "egocentric_images": ["observed.png"]}

    def test_original_kss_and_avu_resolve_observed_object(self):
        client = RecordedReplies("1", "Object 1\nThe observed image matches.")
        result = MSGNavDecision(client).decide(self.obs, self.root)
        self.assertEqual(result["goal_xy_m"], [3., 2.])
        self.assertEqual(result["api_call_count"], 2)
        self.assertEqual(result["compressed_snapshot_count"], 1)
        self.assertFalse(result["task_success_verified"])
        self.assertTrue(any(len(item) == 2 for item in client.requests[1][1]))

    def test_original_single_frontier_exploration_branch(self):
        result = MSGNavDecision(RecordedReplies("1", "Continue Exploration")).decide(self.obs, self.root)
        self.assertEqual(result["choice_type"], "frontier")
        self.assertEqual(result["goal_xy_m"], [1., 0.])
        self.assertFalse(result["task_success_verified"])

    def test_object_center_is_not_used_as_drive_target(self):
        del self.obs["objects"][0]["approach_xy_m"]
        result = MSGNavDecision(RecordedReplies("1", "Object 1")).decide(self.obs, self.root)
        self.assertFalse(result["goal_resolved"])
        self.assertIsNone(result["goal_xy_m"])

    def test_egocentric_image_needs_depth_localization(self):
        result = MSGNavDecision(RecordedReplies("1", "Image 1, jacket")).decide(self.obs, self.root)
        self.assertFalse(result["goal_resolved"])
        self.assertEqual(result["evidence"], {"egocentric_index": 0})

    def test_invalid_decisions_stop_instead_of_guessing(self):
        client = RecordedReplies("1", "Object 999", "Object 999", "Object 999")
        with self.assertRaisesRegex(RuntimeError, "no valid decision"):
            MSGNavDecision(client).decide(self.obs, self.root)

    def test_hidden_stage_objects_are_not_an_allowed_input_source(self):
        self.obs["source"] = "hidden_stage_ground_truth"
        client = RecordedReplies()
        with self.assertRaises(ValueError):
            MSGNavDecision(client).decide(self.obs, self.root)
        self.assertEqual(len(client.requests), 0)

    def test_pointgoal_input_uses_seven_images_and_six_scaled_positions(self):
        self.obs["pointgoal"] = {"history": [{"image": "observed.png", "xy_m": [i*.1, 0.], "capture_time_s": float(i+1)} for i in range(4)],
                                 "tri_view": {key: "observed.png" for key in ["left", "front", "right"]}}
        request = prepare_request(self.obs, [3., -1.5], self.root)
        self.assertEqual(len(request["images"]), 7)
        self.assertEqual(len(request["input_positions_scaled"]), 6)
        self.assertEqual(request["input_positions_scaled"][-1], [5., 10.])
        self.assertEqual(request["content_text"].count("<image>"), 7)
        self.assertIn("<|NAV|>", request["content_text"])

    def test_pointgoal_rejects_fabricated_time_order(self):
        self.obs["pointgoal"] = {"history": [{"image": "observed.png", "xy_m": [0., 0.], "capture_time_s": 1.}] * 4,
                                 "tri_view": {key: "observed.png" for key in ["left", "front", "right"]}}
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            prepare_request(self.obs, [3., 2.], self.root)


class FrameAndExecutionTests(unittest.TestCase):
    def setUp(self):
        self.robot = json.loads((PROJECT / "configs/robot_v1.json").read_text())
        self.cfg = json.loads((PROJECT / "configs/crossing_experiment.json").read_text())
        class DryField:
            water_z = 0.
            def mesh_height(self, x, y):
                return 1.
        self.bridge = PhysicalNavigationBridge(self.robot, DryField(), self.cfg)
        self.row = {"time_s": 10., "x_m": 0., "y_m": 0., "yaw_rad": 0.,
                    "vx_m_s": 0., "vy_m_s": 0., "wz_rad_s": 0., "buoyancy_n": 0.,
                    "left_normal_n": 235.44, "right_normal_n": 235.44}
        self.result = {"source": "OmniNav_Slowfast_dedicated_head", "capture_time_s": 10.,
                       "anchor_pose_xyyaw": [0., 0., 0.],
                       "waypoints_normalized": [[2., 0.]] * 5, "turn_angles_rad": [0.] * 5}

    def test_right_forward_axes_and_rotation(self):
        self.assertEqual(omni_local_point([2., -1.], [0., 0., 0.]), [1., 2.])
        self.assertEqual(omni_local_point([0., 2.], [0., 0., math.pi/2]), [0., 2.])
        positions = pointgoal_positions([[0., 0.]] * 4, [0., 0.], [3., -1.5], [0., 0., 0.])
        self.assertEqual(positions[-1], [5., 10.])

    def test_native_radius_and_angle_are_both_used(self):
        goal = native_head_first_step([[3., 4.]] * 5, [math.pi/2] * 5, [5., 7., 0.])
        self.assertAlmostEqual(goal[0], 5.)
        self.assertAlmostEqual(goal[1], 8.5)

    def test_forward_head_output_becomes_track_commands(self):
        self.bridge.accept_head(self.result, 10.)
        result = self.bridge.tick(self.row, .05)
        self.assertEqual(result["status"], "tracking")
        self.assertGreater(min(result["command"]["tracks"]), 0.)
        self.assertEqual(result["command"]["thrusters"], [0., 0.])
        self.assertEqual(self.row["x_m"], 0.)

    def test_left_turn_drives_right_track_faster(self):
        self.result["turn_angles_rad"] = [math.pi/3] * 5
        self.bridge.accept_head(self.result, 10.)
        result = self.bridge.tick(self.row, .05)
        left, right = result["command"]["tracks"]
        self.assertGreater(right, left)

    def test_five_point_polyline_is_tracked_instead_of_stopping_at_first_point(self):
        result = dict(self.result)
        result["waypoints_normalized"] = [[0.667, 0.], [1.333, 0.], [2.0, 0.], [2.667, 0.], [3.333, 0.]]
        result["turn_angles_rad"] = [0.] * 5
        plan = self.bridge.accept_head(result, 10.)
        self.assertAlmostEqual(plan["path_length_m"], 1.0, delta=.01)
        self.assertEqual(len(plan["path_xy_m"]), 6)
        row = dict(self.row, x_m=.05)
        first = self.bridge.tick(row, .05)
        self.assertEqual(first["status"], "tracking")
        self.assertGreater(first["target_xy_m"][0], row["x_m"])
        self.assertGreater(min(first["command"]["tracks"]), 0.)
        row = dict(row, x_m=.95)
        end = self.bridge.tick(row, .05)
        self.assertEqual(end["status"], "polyline_reached")
        self.assertEqual(end["command"]["tracks"], [0., 0.])

    def test_polyline_segment_beyond_limit_is_rejected(self):
        result = dict(self.result)
        result["waypoints_normalized"] = [[0.667, 0.], [3.0, 0.], [3.1, 0.], [3.2, 0.], [3.3, 0.]]
        result["turn_angles_rad"] = [0.] * 5
        with self.assertRaisesRegex(ValueError, "segment"):
            self.bridge.accept_head(result, 10.)

    def test_truncate_plan_keeps_only_the_safe_prefix(self):
        result = dict(self.result)
        result["waypoints_normalized"] = [[0.667, 0.], [1.333, 0.], [2.0, 0.], [2.667, 0.], [3.333, 0.]]
        result["turn_angles_rad"] = [0.] * 5
        plan = self.bridge.accept_head(result, 10.)
        truncated = self.bridge.truncate_plan(plan["path_xy_m"][:3])
        self.assertAlmostEqual(truncated["path_length_m"], .4, delta=.01)
        self.assertEqual(len(truncated["path_xy_m"]), 3)
        row = dict(self.row, x_m=.1)
        tracking = self.bridge.tick(row, .05)
        self.assertEqual(tracking["status"], "tracking")
        row = dict(row, x_m=.38)
        done = self.bridge.tick(row, .05)
        self.assertEqual(done["status"], "polyline_reached")

    def test_safe_polyline_prefix_stops_at_first_blocked_segment(self):
        prefix, length = safe_polyline_prefix(
            [[0., 0.], [.2, 0.], [.4, 0.], [.6, 0.]],
            lambda start, end: end[0] <= .3)
        self.assertEqual(prefix, [[0., 0.], [.2, 0.]])
        self.assertAlmostEqual(length, .2, places=6)

    def test_near_zero_translation_still_executes_predicted_turn(self):
        self.result['waypoints_normalized'] = [[0., .01]]*5
        self.result['turn_angles_rad'] = [.45]*5
        self.bridge.accept_head(self.result, 10.)
        result = self.bridge.tick(self.row, .05)
        self.assertEqual(result['status'], 'heading_alignment')
        left, right = result['command']['tracks']
        self.assertLess(left, 0.)
        self.assertGreater(right, 0.)
        self.row['yaw_rad'] = .45
        result = self.bridge.tick(self.row, .05)
        self.assertEqual(result['status'], 'short_waypoint_reached')
        self.assertEqual(result['command']['tracks'], [0., 0.])

    def test_expiry_zeroes_commands_instead_of_replaying_old_plan(self):
        self.bridge.accept_head(self.result, 10.)
        self.row["time_s"] = 12.01
        result = self.bridge.tick(self.row, .05)
        self.assertEqual(result["status"], "expired_plan")
        self.assertEqual(result["command"]["tracks"], [0., 0.])
        self.assertEqual(result["command"]["thrusters"], [0., 0.])

    def test_stale_out_of_order_and_generic_chat_are_rejected(self):
        with self.assertRaises(ValueError):
            self.bridge.accept_head(self.result, 13.)
        self.bridge.accept_head(self.result, 10.)
        with self.assertRaises(ValueError):
            self.bridge.accept_head(self.result, 10.5)
        self.result["source"] = "generic_qwen_chat"
        with self.assertRaises(ValueError):
            self.bridge.accept_head(self.result, 10.5)

    def test_nonfinite_or_huge_prediction_is_rejected(self):
        self.result["waypoints_normalized"][0] = [float("nan"), 0.]
        with self.assertRaises(ValueError):
            self.bridge.accept_head(self.result, 10.)
        self.result["waypoints_normalized"][0] = [100., 0.]
        with self.assertRaises(ValueError):
            self.bridge.accept_head(self.result, 10.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
