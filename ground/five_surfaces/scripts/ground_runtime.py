"""Native PhysX runtime shared by GUI and renderer-free turning batches."""
import json
import math
from pathlib import Path
import traceback

import omni.timeline
from pxr import UsdGeom

from ground_environment import FiveSurfaceEnvironment
from ground_telemetry import ActiveSimulationClock, GroundTelemetry
from robot_runtime import RobotRuntime
from scm_terrain import SCMTerrain, SCMTrackForces
from track_motion import CirculatingTracks


class GroundRuntime(RobotRuntime):
    def __init__(self, project, config, scm_enabled=None, terrain=None):
        self.scm_settings = json.loads((Path(project)/'configs/scm.json').read_text())
        self.scm_enabled = self.scm_settings['enabled_by_default'] if scm_enabled is None else scm_enabled
        self.terrain = None
        self.track_settings = json.loads((Path(project)/'configs/tracks.json').read_text())
        self.track_motion = None
        self.ground_config = config
        self.running_route = False
        self.turn_trial = None
        self.route_reason = None
        self.visited = set()
        self.next_control_s = 0.
        self.braking_since = None
        self.route_start_s = 0.
        self.performance = ActiveSimulationClock()
        super().__init__(project, 'land')
        self.environment = FiveSurfaceEnvironment(config)
        collider_points = [list(UsdGeom.Mesh(self.stage.GetPrimAtPath(
            '/World/Robot/Tracks/' + side + '/Collider')).GetPointsAttr().Get())
            for side in ('Left', 'Right')]
        self.telemetry = GroundTelemetry(self.cfg, config, collider_points)
        if self.scm_enabled:
            self.terrain = terrain if terrain is not None else SCMTerrain(config,self.scm_settings,self.cfg,collider_points)
            if self.track_settings['enabled']:
                self.track_motion = CirculatingTracks(self.cfg,self.track_settings)
            self.terrain.track_surface = self.track_motion
            self.model = SCMTrackForces(self.cfg,self.terrain)
        self.set_commands('LAND', [0., 0.], [0., 0.])

    def on_timeline(self, event):
        super().on_timeline(event)
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self.performance.resume(self.sim_time)
        elif event.type in (int(omni.timeline.TimelineEventType.PAUSE), int(omni.timeline.TimelineEventType.STOP)):
            self.performance.pause(self.sim_time)

    def begin_route(self):
        self.visited.clear()
        self.route_reason = None
        self.running_route = True
        self.route_start_s = self.sim_time
        self.braking_since = None

    def finish_route(self, reason):
        self.zero()
        self.running_route = False
        self.route_reason = reason
        self.timeline.pause()

    def on_step(self, dt):
        if self.timeline.is_playing():
            self.performance.resume(self.sim_time)
        before = self.sim_time
        super().on_step(dt)
        if before == self.sim_time or self.error:
            return
        row = self.latest
        try:
            row.update(self.telemetry.measure(row, float(dt)))
            if self.scm_enabled:
                row.update(self.model.soil_metrics)
            row['ground_model'] = 'scm' if self.scm_enabled else 'rigid'
            row.update(self.performance.sample(self.sim_time))
        except Exception:
            self.fail('ground_telemetry', traceback.format_exc())
            return
        tile = self.environment.surface_at(row['x_m'], row['y_m'])
        row['surface'] = tile['id'] if tile else 'Outside'
        row['surface_dynamic_friction'] = tile['dynamic_friction'] if tile else 0.
        if tile and row['left_normal_n']+row['right_normal_n'] > 10.:
            self.visited.add(tile['id'])
        if self.turn_trial is not None:
            row.update(self.turn_trial.update(row))
            self.set_commands('LAND', self.turn_trial.commands, [0., 0.])
            if self.turn_trial.reason:
                self.zero()
                self.timeline.pause()
        if not self.running_route or self.sim_time < self.next_control_s:
            return
        self.next_control_s = self.sim_time+.05
        elapsed = self.sim_time-self.route_start_s
        if elapsed > 90 or not tile or abs(row['roll_deg']) > 30 or abs(row['pitch_deg']) > 30:
            self.finish_route('safety_stop')
            return
        if elapsed < 1.:
            self.zero()
            return
        if row['x_m'] >= self.ground_config['goal_x_m'] or self.braking_since is not None:
            self.zero()
            if self.braking_since is None:
                self.braking_since = self.sim_time
            if math.hypot(row['vx_m_s'],row['vy_m_s']) < .05 and self.sim_time-self.braking_since > .6:
                self.finish_route('completed')
            return
        yaw = math.atan2(math.sin(row['yaw_rad']), math.cos(row['yaw_rad']))
        yaw_rate = max(-.45, min(.45, -1.4*yaw-.18*row['y_m']))
        speed = self.ground_config['drive_speed_m_s']
        half = self.cfg['tracks']['center_separation_m']/2
        self.set_commands('LAND', [speed-yaw_rate*half, speed+yaw_rate*half], [0.,0.])

