"""Turning completion geometry, stress aggregation and comparison conventions."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'scripts'))
from turning_experiments import TurningTrial, combined_slip_rms
from test_scm import make_terrain


class TurningTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((PROJECT/'configs/turning.json').read_text())
        self.tile = json.loads((PROJECT/'configs/five_surfaces.json').read_text())['surfaces'][2]

    def row(self, t, x=0., y=0., yaw=0.):
        return dict(time_s=t, x_m=x, y_m=y, yaw_rad=yaw, wz_rad_s=0.,
                    roll_deg=0., pitch_deg=0., native_track_normal_n=0.)

    def test_spin_commands_opposed_belts_and_unwraps_yaw(self):
        trial = TurningTrial('spin', self.tile, self.settings, .7)
        trial.update(self.row(0.))
        trial.update(self.row(2.1))
        self.assertLess(trial.commands[0], 0)
        self.assertGreater(trial.commands[1], 0)
        for i, angle in enumerate(np.linspace(0., 2*math.pi+.01, 100)):
            trial.update(self.row(2.2+i*.1, yaw=math.atan2(math.sin(angle), math.cos(angle))))
        self.assertEqual(trial.reason, 'completed')
        self.assertEqual(trial.commands, (0., 0.))

    def test_orbit_uses_polar_angle_not_body_yaw(self):
        trial = TurningTrial('orbit', self.tile, self.settings, .7)
        radius = trial.radius
        trial.update(self.row(0., y=-radius))
        trial.update(self.row(2.1, y=-radius))
        for i, yaw in enumerate(np.linspace(0., 2*math.pi+.01, 100)):
            trial.update(self.row(2.2+i*.1, y=-radius, yaw=yaw))
        self.assertAlmostEqual(trial.progress, 0.)
        self.assertIsNone(trial.reason)
        for i, angle in enumerate(np.linspace(-math.pi/2, 3*math.pi/2+.01, 100)):
            trial.update(self.row(13.+i*.1, radius*math.cos(angle), radius*math.sin(angle)))
        self.assertEqual(trial.reason, 'completed')

    def test_outside_material_and_bedrock_stop(self):
        trial = TurningTrial('spin', self.tile, self.settings, .7)
        trial.update(self.row(0., x=1.8))
        self.assertEqual(trial.reason, 'boundary_stop')
        trial = TurningTrial('spin', self.tile, self.settings, .7)
        row = self.row(0.)
        row['native_track_normal_n'] = 10.
        trial.update(row)
        self.assertEqual(trial.reason, 'bedrock_contact')

    def test_counterrotation_slip_does_not_cancel(self):
        row = dict(left_slip_ratio=-.5, right_slip_ratio=.5,
                   left_normal_n=200., right_normal_n=200., left_slip_valid=1, right_slip_valid=1)
        self.assertAlmostEqual(combined_slip_rms(row), .5)
        row['left_slip_valid'] = row['right_slip_valid'] = 0
        self.assertTrue(math.isnan(combined_slip_rms(row)))

    def test_pressure_integral_and_shear_magnitude_statistics(self):
        terrain = make_terrain()
        force, _, stats = terrain.compute([0,0,.20], [0,0,0,1], np.zeros(6), [-.4,.4], 1/120)
        self.assertAlmostEqual(stats['soil_pressure_mean_pa']*stats['scm_contact_area_m2'], force[2])
        # Opposed traction nearly cancels, but local shear intensity must not.
        self.assertLess(abs(force[0]), 1.)
        self.assertGreater(stats['soil_shear_stress_mean_pa'], 100.)
        self.assertGreaterEqual(stats['soil_shear_stress_max_pa'], stats['soil_shear_stress_mean_pa'])
        _, _, air = terrain.compute([0,0,.5], [0,0,0,1], np.zeros(6), [0.,0.], 1/120)
        self.assertTrue(math.isnan(air['soil_pressure_mean_pa']))
        self.assertTrue(math.isnan(air['left_shear_stress_mean_pa']))
        self.assertEqual(set(stats), set(air))


if __name__ == '__main__':
    unittest.main()
