import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from ground_telemetry import ActiveSimulationClock, GroundTelemetry, metric_text


def state(**changes):
    row = dict(x_m=-8.5, y_m=0., z_m=.22, qx=0., qy=0., qz=0., qw=1.,
               vx_m_s=0., vy_m_s=0., vz_m_s=0., wx_rad_s=0., wy_rad_s=0., wz_rad_s=0.,
               left_belt_m_s=0., right_belt_m_s=0., left_normal_n=235., right_normal_n=235.)
    row.update(changes)
    return row


class TrackTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.robot = json.loads((PROJECT / 'configs/robot_v1.json').read_text())
        self.ground = json.loads((PROJECT / 'configs/five_surfaces.json').read_text())
        self.points = [[[x, y, -.22] for x in (-.3, .3) for y in (center-.08, center+.08)]
                       for center in (.35, -.35)]
        self.metrics = GroundTelemetry(self.robot, self.ground, self.points)

    def test_pure_travel_and_traction_slip(self):
        result = self.metrics.measure(state(vx_m_s=.4, left_belt_m_s=.4, right_belt_m_s=.8), .1)
        self.assertAlmostEqual(result['left_slip_ratio'], 0.)
        self.assertAlmostEqual(result['right_slip_ratio'], .5)
        self.assertAlmostEqual(result['right_slip_distance_m'], .04)

    def test_reverse_braking_and_counter_rotation_keep_sign(self):
        reverse = self.metrics.measure(state(vx_m_s=-.4, left_belt_m_s=-.8), .1)
        self.assertAlmostEqual(reverse['left_slip_ratio'], -.5)
        braking = self.metrics.measure(state(vx_m_s=.4), .1)
        self.assertAlmostEqual(braking['left_slip_ratio'], -1.)
        counter = self.metrics.measure(state(vx_m_s=-.4, left_belt_m_s=.4), .1)
        self.assertAlmostEqual(counter['left_slip_ratio'], 2.)

    def test_turning_uses_local_velocity_and_lateral_rms(self):
        result = self.metrics.measure(state(wz_rad_s=1., left_belt_m_s=-.35, right_belt_m_s=.35), .1)
        self.assertAlmostEqual(result['left_patch_vx_m_s'], -.35)
        self.assertAlmostEqual(result['right_patch_vx_m_s'], .35)
        self.assertAlmostEqual(result['left_slip_ratio'], 0.)
        # Opposite transverse velocities at the front and rear must not cancel.
        self.assertAlmostEqual(result['left_lateral_slip_rms_m_s'], .3)

    def test_yaw_rotates_velocity_to_body_forward(self):
        q = math.sqrt(.5)
        result = self.metrics.measure(state(qz=q, qw=q, vx_m_s=0., vy_m_s=.4,
                                            left_belt_m_s=.4), .1)
        self.assertAlmostEqual(result['left_slip_ratio'], 0.)
        self.assertAlmostEqual(result['left_lateral_slip_rms_m_s'], 0.)

    def test_low_speed_and_airborne_are_invalid(self):
        stopped = self.metrics.measure(state(), .1)
        self.assertEqual(stopped['left_slip_valid'], 0)
        self.assertTrue(math.isnan(stopped['left_slip_ratio']))
        airborne = self.metrics.measure(state(left_normal_n=0., left_belt_m_s=1., z_m=.5), .1)
        self.assertEqual(airborne['left_ground_contact'], 0)
        self.assertTrue(math.isnan(airborne['left_slip_ratio']))
        self.assertEqual(airborne['left_slip_distance_m'], 0.)

    def test_initial_drop_is_clearance_not_sinkage(self):
        initial = self.metrics.measure(state(z_m=.24), .1)
        landed = self.metrics.measure(state(z_m=.22), .1)
        sunk = self.metrics.measure(state(z_m=.19), .1)
        self.assertAlmostEqual(initial['left_ground_clearance_m'], .02)
        self.assertEqual(initial['left_sinkage_m'], 0.)
        self.assertAlmostEqual(landed['left_sinkage_m'], 0.)
        self.assertAlmostEqual(sunk['left_sinkage_m'], .03)

    def test_sinkage_accounts_for_roll_and_surface_bounds(self):
        angle = .1
        result = self.metrics.measure(state(qx=math.sin(angle/2), qw=math.cos(angle/2)), .1)
        transformed = np.array(self.points[1])
        minimum_z = np.min(transformed[:, 1]*math.sin(angle) + transformed[:, 2]*math.cos(angle)) + .22
        self.assertAlmostEqual(result['right_sinkage_m'], max(0., -minimum_z))
        outside = self.metrics.measure(state(x_m=30.), .1)
        self.assertEqual(outside['left_sinkage_valid'], 0)
        self.assertTrue(math.isnan(outside['left_sinkage_m']))


class PerformanceTests(unittest.TestCase):
    def test_pause_excluded_and_window_restarts(self):
        now = [100.]
        meter = ActiveSimulationClock(clock=lambda: now[0])
        meter.resume(0.)
        now[0] += .5
        first = meter.sample(1.)
        self.assertAlmostEqual(first['rtf_average'], 2.)
        self.assertAlmostEqual(first['rtf_window'], 2.)
        meter.pause(1.)
        now[0] += 1000.
        meter.resume(1.)
        now[0] += .5
        resumed = meter.sample(2.)
        self.assertAlmostEqual(resumed['active_wall_time_s'], 1.)
        self.assertAlmostEqual(resumed['rtf_average'], 2.)
        self.assertAlmostEqual(resumed['rtf_window'], 2.)

    def test_window_tracks_recent_speed_instead_of_lifetime(self):
        now = [0.]
        meter = ActiveSimulationClock(clock=lambda: now[0])
        meter.resume(0.)
        for i in range(1, 21):
            now[0] = i*.1
            sim = i*.2 if i <= 10 else 2.+(i-10)*.05
            result = meter.sample(sim)
        self.assertAlmostEqual(result['rtf_window'], .5)
        self.assertAlmostEqual(result['rtf_average'], 1.25)
        self.assertLessEqual(len(meter.samples), 12)

    def test_display_marks_unavailable_metrics(self):
        row = state()
        config = json.loads((PROJECT/'configs/robot_v1.json').read_text())
        ground = json.loads((PROJECT/'configs/five_surfaces.json').read_text())
        row.update(GroundTelemetry(config, ground, [np.array([[0., .35, -.22]]),
                                                   np.array([[0., -.35, -.22]])]).measure(row, .01))
        row.update(ActiveSimulationClock(clock=lambda: 0.).sample(0.))
        row['time_s'] = 0.
        text = metric_text(row)
        self.assertIn('N/A / N/A', text)
        self.assertIn('warming up', text)
        self.assertIn('0.000 / 0.000 mm', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
