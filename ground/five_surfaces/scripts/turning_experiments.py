"""Kit-independent circle controller and experiment metric definitions.

Only commands belt speeds: vehicle pose is never prescribed during a trial.
"""
import math

import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class TurningTrial:
    def __init__(self, mode, tile, settings, separation, max_belt_speed=1.):
        if mode not in ('spin', 'orbit'):
            raise ValueError('Turning mode must be spin or orbit')
        self.mode, self.tile, self.settings = mode, tile, dict(settings)
        self.separation, self.max_belt_speed = separation, max_belt_speed
        self.center = np.array([(tile['x_min_m']+tile['x_max_m'])/2,
                                (tile['y_min_m']+tile['y_max_m'])/2])
        self.radius = 0. if mode == 'spin' else settings['orbit_radius_m']
        if settings['revolutions'] <= 0 or settings['max_trial_s'] <= settings['settle_s']:
            raise ValueError('Positive revolutions and sufficient trial duration required')
        if mode == 'orbit' and (self.radius <= 0 or settings['orbit_speed_m_s'] <= 0):
            raise ValueError('Orbit radius and speed must be positive')
        margin = settings['boundary_margin_m']
        if self.radius+margin >= min(tile['x_max_m']-tile['x_min_m'],
                                     tile['y_max_m']-tile['y_min_m'])/2:
            raise ValueError('Circle and vehicle margin must fit inside one surface')
        self.start_position = [float(self.center[0]), float(self.center[1]-self.radius), .24]
        self.started = None
        self.previous_angle = None
        self.progress = 0.
        self.next_control = 0.
        self.last_control = None
        self.integral = 0.
        self.commands = (0., 0.)
        self.reason = None

    def update(self, row):
        t = row['time_s']
        if self.started is None:
            self.started = t
        elapsed = t-self.started
        xy = np.array([row['x_m'], row['y_m']])
        offset = xy-self.center
        distance = float(np.linalg.norm(offset))
        angle = row['yaw_rad'] if self.mode == 'spin' else math.atan2(offset[1], offset[0])
        running = elapsed >= self.settings['settle_s']
        if running and self.previous_angle is not None:
            self.progress += wrap(angle-self.previous_angle)
        self.previous_angle = angle if running else None
        margin = self.settings['boundary_margin_m']
        inside = (self.tile['x_min_m']+margin < xy[0] < self.tile['x_max_m']-margin and
                  self.tile['y_min_m']+margin < xy[1] < self.tile['y_max_m']-margin)
        if not inside:
            self.reason = 'boundary_stop'
        elif abs(row['roll_deg']) > 25 or abs(row['pitch_deg']) > 25:
            self.reason = 'attitude_stop'
        elif row.get('native_track_normal_n', 0.) > 5.:
            self.reason = 'bedrock_contact'
        elif self.mode == 'orbit' and running and distance < .2*self.radius:
            self.reason = 'orbit_center_crossed'
        elif self.progress >= 2*math.pi*self.settings['revolutions']:
            self.reason = 'completed'
        elif elapsed >= self.settings['max_trial_s']:
            self.reason = 'timeout'
        if not running or self.reason:
            self.commands = (0., 0.)
        elif t >= self.next_control:
            self.next_control = t+self.settings['control_period_s']
            yaw = row['yaw_rad']
            if self.mode == 'spin':
                forward = np.array([math.cos(yaw), math.sin(yaw)])
                speed = float(np.clip(-self.settings['spin_position_gain']*offset @ forward, -.12, .12))
                target_rate = self.settings['spin_yaw_rate_rad_s']
            else:
                desired_heading = angle+math.pi/2+math.atan2(
                    self.settings['radial_gain']*(distance-self.radius),
                    self.settings['orbit_speed_m_s'])
                error = wrap(desired_heading-yaw)
                speed = self.settings['orbit_speed_m_s']*max(.3, math.cos(error))
                target_rate = speed/self.radius+self.settings['heading_gain']*error
            rate_error = target_rate-row['wz_rad_s']
            dt = 0. if self.last_control is None else t-self.last_control
            self.last_control = t
            self.integral = float(np.clip(self.integral+rate_error*dt, -1., 1.))
            omega = target_rate+self.settings['yaw_rate_gain']*rate_error+self.settings['yaw_integral_gain']*self.integral
            commands = np.array([speed-omega*self.separation/2, speed+omega*self.separation/2])
            # Scale both together to preserve requested curvature at saturation.
            commands /= max(1., float(np.max(np.abs(commands)))/self.max_belt_speed)
            self.commands = tuple(float(v) for v in commands)
        return dict(turn_mode=self.mode, turn_surface=self.tile['id'],
                    turn_phase='settle' if not running else 'turn',
                    turn_elapsed_s=elapsed, turn_time_s=max(0., elapsed-self.settings['settle_s']),
                    turn_center_x_m=float(self.center[0]), turn_center_y_m=float(self.center[1]),
                    turn_target_radius_m=self.radius, turn_actual_radius_m=distance,
                    turn_radius_error_m=distance-self.radius,
                    turn_progress_rad=self.progress, turn_result=self.reason or 'running')


def combined_slip_rms(row):
    """Load-weighted RMS of signed side slip ratios; no spin cancellation."""
    numerator, denominator = 0., 0.
    for side in ('left', 'right'):
        ratio, load = row[side+'_slip_ratio'], row[side+'_normal_n']
        if row[side+'_slip_valid'] and math.isfinite(ratio) and load > 0:
            numerator += load*ratio*ratio
            denominator += load
    return math.sqrt(numerator/denominator) if denominator else math.nan


METRIC_DEFINITIONS = {
    'pressure': 'Dynamic normal pressure after damping/clamping, Pa; mean=sum(p_i*A_i)/sum(A_i).',
    'shear_stress': 'Final bounded XY shear stress magnitude, Pa; mean=sum(norm(tau_i)*A_i)/sum(A_i). Not norm of the vector mean.',
    'stress_max': 'Maximum sampled contact-cell value; not an area average.',
    'stress_scope': 'Compressed contact coverage, including zero-pressure cells if geometrically compressed; no contact gives NaN.',
    'comparison_slip': 'Normal-load-weighted RMS of valid left/right signed slip ratios; plotted in percent. Raw signed values remain in CSV.',
    'comparison_sinkage': 'Maximum total sinkage of the two sides, mm; not permanent rut depth.',
    'time_axis': 'Simulation seconds since end of settle phase; settle rows excluded from comparison plots.',
    'completion': 'Spin: accumulated unwrapped chassis yaw. Orbit: accumulated polar angle about the specified fixed center. Positive CCW revolutions.',
    'reset': 'Each material/mode trial resets vehicle, soil and shear history. Same controller and target settings across materials.',
}
