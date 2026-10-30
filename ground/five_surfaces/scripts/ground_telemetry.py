"""Observation only: equivalent-track slip, geometric sinkage and active-time RTF.

This module has no Kit dependencies and never modifies the force model.
"""
from collections import deque
import math
import time

import numpy as np

from amphibian_dynamics import quaternion_matrix_xyzw


class ActiveSimulationClock:
    """Wall time includes rendering/UI between steps, excluding timeline pauses."""

    def __init__(self, clock=time.perf_counter, window_s=1.0):
        self.clock = clock
        self.window_s = window_s
        self.wall_s = 0.0
        self.sim_s = 0.0
        self.anchor = None
        self.samples = deque()

    def resume(self, sim_time):
        if self.anchor is None:
            self.anchor = (self.clock(), sim_time)
            self.samples.clear()
            self.samples.append((self.wall_s, self.sim_s))

    def pause(self, sim_time):
        if self.anchor is not None:
            now = self.clock()
            self.wall_s += max(0.0, now - self.anchor[0])
            self.sim_s += max(0.0, sim_time - self.anchor[1])
            self.anchor = None
            self.samples.clear()

    def sample(self, sim_time):
        # Also supports attaching controls while the timeline is already playing.
        self.resume(sim_time)
        wall = self.wall_s + max(0.0, self.clock() - self.anchor[0])
        sim = self.sim_s + max(0.0, sim_time - self.anchor[1])
        self.samples.append((wall, sim))
        while len(self.samples) > 2 and self.samples[1][0] <= wall - self.window_s:
            self.samples.popleft()
        elapsed = wall - self.samples[0][0]
        return {
            'active_wall_time_s': wall,
            'rtf_sim_time_s': sim,
            'rtf_average': sim / wall if wall > 1e-6 else math.nan,
            'rtf_window': (sim - self.samples[0][1]) / elapsed if elapsed >= .1 else math.nan,
            'rtf_window_wall_s': elapsed,
        }


class GroundTelemetry:
    SPEED_EPSILON_M_S = .05
    CONTACT_THRESHOLD_N = .5

    def __init__(self, robot_config, ground_config, collider_points):
        self.tracks = robot_config['tracks']
        self.ground = ground_config
        # Points are the actual collider vertices in the robot body frame.
        self.collider_points = [np.asarray(points, dtype=float) for points in collider_points]
        self.slip_distance = [0.0, 0.0]

    def height_at(self, x, y):
        for tile in self.ground['surfaces']:
            if tile['x_min_m'] <= x <= tile['x_max_m'] and tile['y_min_m'] <= y <= tile['y_max_m']:
                return self.ground['top_z_m']
        return None

    def measure(self, row, dt):
        rotation = quaternion_matrix_xyzw([row[k] for k in ('qx', 'qy', 'qz', 'qw')])
        position = np.array([row[k] for k in ('x_m', 'y_m', 'z_m')])
        linear = np.array([row[k] for k in ('vx_m_s', 'vy_m_s', 'vz_m_s')])
        angular = np.array([row[k] for k in ('wx_rad_s', 'wy_rad_s', 'wz_rad_s')])
        # All five current surfaces are horizontal. Match the force model's
        # projection of the body forward axis onto the contact tangent plane.
        forward = rotation[:, 0].copy()
        forward[2] = 0.0
        forward_norm = np.linalg.norm(forward)
        forward /= max(1e-9, forward_norm)
        lateral = np.cross([0.0, 0.0, 1.0], forward)
        result = {}
        for side, name in enumerate(('left', 'right')):
            y = self.tracks['center_separation_m'] / 2 * (1 if side == 0 else -1)
            patches = np.array([[x, y, self.tracks['center_z_m'] - self.tracks['height_m'] / 2]
                                for x in self.tracks['patch_x_m']])
            levers = patches @ rotation.T
            velocities = linear + np.cross(angular, levers)
            vx = float(np.mean(velocities @ forward))
            lateral_rms = float(np.sqrt(np.mean((velocities @ lateral) ** 2)))
            belt = row[name + '_belt_m_s']
            slip_speed = belt - vx
            in_contact = row[name + '_normal_n'] > self.CONTACT_THRESHOLD_N
            valid = in_contact and forward_norm > 1e-6 and max(abs(belt), abs(vx)) >= self.SPEED_EPSILON_M_S
            ratio = slip_speed / max(abs(belt), abs(vx), self.SPEED_EPSILON_M_S) if valid else math.nan
            if in_contact:
                self.slip_distance[side] += abs(slip_speed) * dt
            world_points = self.collider_points[side] @ rotation.T + position
            clearances = [float(point[2] - height) for point in world_points
                          if (height := self.height_at(point[0], point[1])) is not None]
            clearance = min(clearances) if clearances else math.nan
            result.update({
                name + '_ground_contact': int(in_contact),
                name + '_patch_vx_m_s': vx,
                name + '_slip_velocity_m_s': slip_speed,
                name + '_slip_ratio': ratio,
                name + '_slip_valid': int(valid),
                name + '_lateral_slip_rms_m_s': lateral_rms,
                name + '_slip_distance_m': self.slip_distance[side],
                name + '_ground_clearance_m': clearance,
                name + '_sinkage_m': max(0.0, -clearance) if clearances else math.nan,
                name + '_sinkage_valid': int(bool(clearances)),
            })
        return result


def metric_text(row):
    def pair(key, scale=1.0, suffix='', digits=2):
        values = []
        for side in ('left', 'right'):
            value = row[side + key]
            values.append(f'{value * scale:.{digits}f}{suffix}' if math.isfinite(value) else 'N/A')
        return ' / '.join(values)

    def rtf(key):
        value = row[key]
        return f'{value:.2f}x' if math.isfinite(value) else 'warming up'

    return (f"Slip L / R: {pair('_slip_ratio', 100, '%', 1)}\n"
            f"Slip speed L / R: {pair('_slip_velocity_m_s')} m/s\n"
            f"Lateral RMS L / R: {pair('_lateral_slip_rms_m_s')} m/s\n"
            f"Sinkage L / R: {pair('_sinkage_m', 1000, '', 3)} mm\n"
            f"RTF (~1 s): {rtf('rtf_window')} | avg: {rtf('rtf_average')}\n"
            f"Sim: {row['time_s']:.2f} s | active wall: {row['active_wall_time_s']:.2f} s")


def telemetry_metadata():
    return {
        'schema_version': 1,
        'slip_ratio': '(filtered_belt_speed - mean_local_patch_vx) / max(abs(belt), abs(vx), 0.05)',
        'slip_valid': 'normal load > 0.5 N and speed >= 0.05 m/s; invalid ratios are NaN in CSV and N/A in UI',
        'slip_direction': 'signed body-forward convention, including reverse; counter-rotation may exceed 100%',
        'slip_source': 'equivalent belt model state and native rigid body velocities, including omega cross r',
        'lateral_slip': 'RMS transverse velocity of each side contact patches; m/s',
        'slip_distance': 'integral of abs(longitudinal slip velocity) during contact; m',
        'sinkage': 'max(0, -minimum collider vertex clearance above initial flat ground); m',
        'sinkage_scope': 'geometric diagnostic, including numerical penetration; rigid ground has no soil deformation',
        'clearance': 'signed minimum collider vertex height minus ground height; positive means above surface',
        'rtf': 'simulated elapsed / monotonic active wall elapsed; excludes timeline pauses and pre-play scene load',
        'rtf_window': 'rolling approximately 1 active wall second, restarts on resume; NaN during first 0.1 s',
        'rtf_scope': 'includes rendering, UI and telemetry overhead between physics callbacks; not physics-only speed',
    }
