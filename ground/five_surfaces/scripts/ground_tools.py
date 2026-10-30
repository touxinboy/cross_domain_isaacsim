"""SCM soft-soil controls with an explicit switch to the original rigid baseline."""
import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
import traceback

import carb
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Usd, UsdGeom

from ground_environment import FiveSurfaceEnvironment
from ground_telemetry import ActiveSimulationClock, GroundTelemetry, metric_text, telemetry_metadata
from robot_assets import set_camera
from robot_runtime import RobotRuntime
from scm_terrain import SCMTerrain, SCMTrackForces
from scm_scene import configure_ground, SCMTerrainView
from track_motion import CirculatingTracks
from track_visuals import TrackVisuals


class GroundRuntime(RobotRuntime):
    def __init__(self, project, config, scm_enabled=None, terrain=None):
        self.scm_settings = json.loads((Path(project)/'configs/scm.json').read_text())
        self.scm_enabled = self.scm_settings['enabled_by_default'] if scm_enabled is None else scm_enabled
        self.terrain = None
        self.track_settings = json.loads((Path(project)/'configs/tracks.json').read_text())
        self.track_motion = None
        self.ground_config = config
        self.running_route = False
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


class GroundTools:
    def __init__(self, project, config, scm_enabled=None):
        self.project, self.config = Path(project), config
        self.runtime = GroundRuntime(project, config,scm_enabled)
        configure_ground(self.runtime.stage,config,self.runtime.scm_enabled,self.runtime.scm_settings)
        self.terrain_view = SCMTerrainView(self.runtime.stage,self.runtime.terrain) if self.runtime.scm_enabled else None
        self.track_view = TrackVisuals(self.runtime.stage,self.runtime.track_motion) if self.runtime.track_motion else None
        self.follow_robot = False
        self.saved_route = False
        self.left = ui.SimpleFloatModel(0.)
        self.right = ui.SimpleFloatModel(0.)
        self.window = ui.Window('Five Surfaces - L17 Robot', width=440, height=700)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, margin=10):
                    ui.Label('FIVE CONNECTED SURFACES', style={'font_size':17})
                    ui.Label('20 x 6 m | single chassis, circulating track shoes', height=22)
                    self.mode_label = ui.Label('',height=36,word_wrap=True)
                    ui.Button('Switch SCM / rigid baseline (resets)',height=26,
                              clicked_fn=lambda:asyncio.ensure_future(self.switch_model()))
                    ui.Separator(height=8)
                    ui.Label('LIVE TELEMETRY | slip / sinkage / drive', height=22)
                    self.metrics_label = ui.Label('Slip / sinkage / RTF: waiting for Play', height=190, word_wrap=True)
                    ui.Label('Slip N/A: low speed or no contact', height=20)
                    self.surface_labels = []
                    with ui.CollapsableFrame('Surface materials',collapsed=True,height=0):
                        with ui.VStack():
                            for tile in config['surfaces']:
                                self.surface_labels.append(ui.Label('',height=20))
                    ui.Separator(height=8)
                    with ui.HStack(height=26,spacing=5):
                        ui.Button('Overview',clicked_fn=lambda:self.camera('Overview'))
                        ui.Button('Robot',clicked_fn=lambda:self.camera('Robot'))
                        ui.Button('Side',clicked_fn=lambda:self.camera('Side'))
                    for label, model in [('Left track (m/s)',self.left),('Right track (m/s)',self.right)]:
                        with ui.HStack(height=24):
                            ui.Label(label,width=160)
                            ui.FloatSlider(model=model,min=-1.,max=1.)
                        model.add_value_changed_fn(lambda _:self.manual_commands())
                    with ui.HStack(height=26,spacing=5):
                        ui.Button('Forward',clicked_fn=lambda:self.drive(.45,.45))
                        ui.Button('Reverse',clicked_fn=lambda:self.drive(-.35,-.35))
                        ui.Button('Zero',clicked_fn=lambda:self.drive(0.,0.))
                    with ui.HStack(height=26,spacing=5):
                        ui.Button('Play',clicked_fn=self.runtime.timeline.play)
                        ui.Button('Pause',clicked_fn=self.pause)
                        ui.Button('Reset',clicked_fn=lambda:asyncio.ensure_future(self.reset()))
                    ui.Button('Run across all five surfaces',height=30,clicked_fn=lambda:asyncio.ensure_future(self.run_route()))
                    ui.Button('Reset vehicle / keep ruts',height=26,
                              clicked_fn=lambda:asyncio.ensure_future(self.reset(clear_soil=False)))
                    ui.Button('Export telemetry CSV',height=26,clicked_fn=self.export)
                    self.status_label = ui.Label('Ready on tile 1. Press Forward, then Play.',height=70,word_wrap=True)
        self.update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.on_update)
        self.camera('Overview')
        self.update_mode_label()
        settings=carb.settings.get_settings()
        settings.set_bool('/app/viewport/grid/enabled',False)
        settings.set_int('/persistent/app/viewport/displayOptions',1<<10)

    def camera(self, name):
        self.follow_robot = name == 'Robot'
        get_active_viewport().set_active_camera('/World/Cameras/'+name)

    def manual_commands(self):
        if self.runtime.running_route:
            return
        self.runtime.set_commands('LAND',[self.left.as_float,self.right.as_float],[0.,0.])

    def drive(self, left, right):
        self.runtime.running_route=False
        self.left.set_value(left)
        self.right.set_value(right)
        self.manual_commands()

    def pause(self):
        self.drive(0.,0.)
        self.runtime.timeline.pause()

    def update_mode_label(self):
        self.mode_label.text = ('SCM | moving shoes & grouser contact\nSimplified drivetrain | uncalibrated soil'
                                if self.runtime.scm_enabled else 'Rigid baseline | original force model')
        for tile,label in zip(self.config['surfaces'],self.surface_labels):
            detail = 'SCM soil' if self.runtime.scm_enabled else f"friction {tile['static_friction']:.2f} / {tile['dynamic_friction']:.2f}"
            label.text = f"{tile['index']}  {tile['label']} | {detail}"

    async def switch_model(self):
        await self.reset(scm_enabled=not self.runtime.scm_enabled)

    async def reset(self, clear_soil=True, scm_enabled=None, position=None):
        mode = self.runtime.scm_enabled if scm_enabled is None else scm_enabled
        terrain = self.runtime.terrain if not clear_soil and mode == self.runtime.scm_enabled else None
        self.runtime.running_route=False
        self.runtime.zero()
        self.runtime.timeline.stop()
        self.runtime.shutdown()
        app=omni.kit.app.get_app()
        await app.next_update_async()
        stage=omni.usd.get_context().get_stage()
        with Usd.EditContext(stage,stage.GetRootLayer()):
            robot=stage.GetPrimAtPath('/World/Robot')
            robot.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*(position or self.config['start_position_m'])))
            robot.GetAttribute('xformOp:orient').Set(Gf.Quatf(1))
        self.runtime=GroundRuntime(self.project,self.config,mode,terrain)
        configure_ground(stage,self.config,mode,self.runtime.scm_settings)
        self.terrain_view = SCMTerrainView(stage,self.runtime.terrain) if mode else None
        TrackVisuals.clear(stage)
        self.track_view = TrackVisuals(stage,self.runtime.track_motion) if self.runtime.track_motion else None
        self.update_mode_label()
        self.saved_route=False
        self.metrics_label.text = 'Slip / sinkage / RTF: waiting for Play'
        self.status_label.text = 'Ready on tile 1. Press Forward, then Play.'
        self.left.set_value(0.)
        self.right.set_value(0.)
        await app.next_update_async()

    async def run_route(self):
        await self.reset()
        self.runtime.begin_route()
        self.runtime.timeline.play()

    def export(self):
        if not self.runtime.history:
            return None
        output=self.project/'results/five_surfaces'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True,exist_ok=False)
        self.runtime.save_csv(output/'physics.csv')
        row=self.runtime.latest
        summary={'result':self.runtime.route_reason or 'manual_export',
                 'visited_surfaces':sorted(self.runtime.visited),
                 'physics_samples':len(self.runtime.history),
                 'final_position_m':[row[k] for k in ['x_m','y_m','z_m']],
                 'soil_model':'scm_style_bekker_vector_janosi' if self.runtime.scm_enabled else 'unchanged_L17_rigid_ground',
                 'chassis_mass_and_inertia_unchanged':True,
                 'nominal_track_dimensions_preserved':True,
                 'robot_physics_unchanged':not self.runtime.scm_enabled,
                 'telemetry':telemetry_metadata(),
                 'final_metrics':{key:value for key,value in row.items()
                                  if key.startswith(('rtf_', 'active_wall_', 'left_slip', 'right_slip',
                                                     'left_sinkage', 'right_sinkage', 'left_rut', 'right_rut',
                                                     'left_belt_', 'right_belt_', 'left_sprocket_', 'right_sprocket_',
                                                     'left_grouser_', 'right_grouser_', 'soil_', 'scm_', 'native_track_'))}}
        # JSON uses null for unavailable metrics; CSV retains explicit NaN.
        summary['final_metrics']={key:(None if isinstance(value,float) and not math.isfinite(value) else value)
                                  for key,value in summary['final_metrics'].items()}
        if self.runtime.scm_enabled:
            self.runtime.terrain.save(output/'soil_state.npz')
            summary['scm_settings'] = self.runtime.scm_settings
            summary['track_settings'] = self.runtime.track_settings
            summary['telemetry'].update(sinkage='maximum h0 minus distributed track-envelope ray intersection height in compressed cells',
                sinkage_scope='SCM total geometric sinkage relative to initial soil surface',
                slip_source='normal-force-weighted distributed contact velocities and filtered equivalent belt speed',
                lateral_slip='normal-force-weighted RMS transverse velocity over distributed contact cells; m/s',
                slip_distance='integral of normal-force-weighted abs local longitudinal slip velocity during contact; m',
                rut_depth='plastic sinkage, retained on unload; no bulk soil flow',
                normal_force='sum of SCM normal forces; native_track_normal_n records rigid contact separately')
            if self.runtime.track_motion:
                summary['telemetry'].update(
                    track_geometry='moving plates and grousers follow a fixed loop; no independent link dynamics',
                    sinkage='maximum h0 minus moving shoe/grouser ray intersection height in compressed cells',
                    slip_source='distributed material contact velocities including shoe translation and rotation',
                    belt_phase='integral of filtered belt speed; shared by render geometry and soil contact',
                    sprocket_angle='derived from belt travel and pitch radius, not a physical encoder')
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return output

    def on_update(self, event):
        row=self.runtime.latest
        if self.track_view:
            self.track_view.update(self.runtime.sim_time)
        if self.terrain_view:
            self.terrain_view.update(self.runtime.sim_time,force=not self.runtime.timeline.is_playing())
        if not row:
            return
        self.metrics_label.text = metric_text(row)
        if self.runtime.scm_enabled:
            self.metrics_label.text += (f"\nRuts max: {row['soil_rut_depth_max_m']*1000:.1f} mm"
                f" | cells: {row['scm_contact_cells']}\n"
                f"SCM: {row['scm_step_ms']:.2f} ms | mesh: {self.terrain_view.last_update_ms:.2f} ms")
            if self.runtime.track_motion:
                self.metrics_label.text += (f"\nGrouser cells L/R: {row['left_grouser_contact_cells']} / {row['right_grouser_contact_cells']}")
        if self.follow_robot:
            p=[row[k] for k in ['x_m','y_m','z_m']]
            stage=omni.usd.get_context().get_stage()
            with Usd.EditContext(stage,stage.GetSessionLayer()):
                set_camera(stage,'/World/Cameras/Robot',(p[0]-2.8,p[1]-3.8,p[2]+2.1),(p[0],p[1],p[2]+.1),28)
        self.status_label.text=(f"Surface: {row['surface']}  |  x={row['x_m']:.2f} m\n"
                                f"Speed: {math.hypot(row['vx_m_s'],row['vy_m_s']):.2f} m/s  |  "
                                f"Visited: {len(self.runtime.visited)}/5\n"
                                f"{self.runtime.error or self.runtime.route_reason or ('Auto run' if self.runtime.running_route else 'Manual')}")
        if self.runtime.route_reason and not self.saved_route:
            self.export()
            self.saved_route=True
