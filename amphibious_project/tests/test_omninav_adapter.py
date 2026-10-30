"""CPU tensor fixtures for the inference wrapper; no trained model is loaded."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from navigation_contracts import isaac_world_to_habitat, omni_local_point
from navigation_sources import load_functions
from omninav_pointgoal import OmniNavPointGoal


class Batch(dict):
    def to(self, device):
        return self


class ProcessorFixture:
    def apply_chat_template(self, *args, **kwargs):
        return "<image>" * 7

    def __call__(self, **kwargs):
        return Batch(input_ids=torch.tensor([[1, 2, 3]]))


class HeadFixture:
    device = "cpu"
    def __init__(self):
        self.seen = None
        self.invalid = False

    def forward(self, **kwargs):
        self.seen = kwargs
        waypoints = torch.tensor([[[0., 2.]] * 5])
        if self.invalid:
            waypoints[0, 0, 0] = float("nan")
        return waypoints, torch.zeros((1, 1)), torch.zeros((1, 5)), torch.ones((1, 5))


class OmniAdapterTests(unittest.TestCase):
    def predictor(self):
        # Bypass real checkpoint loading deliberately; tensor transport is tested.
        predictor = OmniNavPointGoal.__new__(OmniNavPointGoal)
        predictor.checkpoint = Path("SYNTHETIC_CPU_FIXTURE_NOT_A_MODEL")
        predictor.model = HeadFixture()
        predictor.processor = ProcessorFixture()
        predictor.torch = torch
        predictor.functions = {"process_vision": lambda images: (images, []), "preprocess": lambda images: images}
        return predictor

    def request(self):
        return {"capture_time_s": 10., "anchor_pose_xyyaw": [4., 2., 0.],
                "images": [None] * 7, "content_text": "CPU transport fixture",
                "input_positions_scaled": [[0., 0.]] * 6}

    def test_forward_arguments_and_native_head_decoding(self):
        predictor = self.predictor()
        result = predictor.predict(self.request())
        self.assertEqual(predictor.model.seen["input_waypoints"].shape, (1, 6, 2))
        self.assertTrue(predictor.model.seen["action_former"])
        self.assertFalse(predictor.model.seen["train"])
        self.assertEqual(predictor.model.seen["train_branch"], ["continue"])
        self.assertAlmostEqual(result["first_step_goal_xy_m"][0], 4.6)
        self.assertAlmostEqual(result["first_step_goal_xy_m"][1], 2.)
        self.assertFalse(result["task_success_verified"])

    def test_nan_head_does_not_become_a_valid_plan(self):
        predictor = self.predictor()
        predictor.model.invalid = True
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            predictor.predict(self.request())

    def test_planar_coordinates_match_original_transform_function(self):
        native = load_functions("third_party/OmniNav/infer_ovon_slowfast/qwen_utils.py",
                                ["transform_to_local_frame"], {"np": np, "R": Rotation})["transform_to_local_frame"]
        matrix = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
        for yaw in [0., math.pi/2, -.8]:
            anchor = [3., -2., yaw]
            world = [5., 1.]
            rotation_i = Rotation.from_euler("z", yaw).as_matrix()
            quaternion_h = Rotation.from_matrix(matrix @ rotation_i @ matrix.T).as_quat()
            local = native(isaac_world_to_habitat(world + [0.]), isaac_world_to_habitat(anchor[:2] + [0.]),
                           dict(zip(["x", "y", "z", "w"], quaternion_h)))
            np.testing.assert_allclose(omni_local_point(world, anchor), [local[0], local[2]], atol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
