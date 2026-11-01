"""SCM soft-soil controls with an explicit switch to the original rigid baseline."""
import asyncio
from datetime import datetime
import math
from pathlib import Path

import carb
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Usd

from ground_telemetry import metric_text
from robot_assets import set_camera
from scm_scene import configure_ground, SCMTerrainView
from track_visuals import TrackVisuals
from ground_runtime import GroundRuntime

class GroundTools:
    def __init__(self, project, config, scm_enabled=None):
        self.project, self.config = Path(project), config
        self.runtime = GroundRuntime(project, config,scm_enabled)
        configure_ground(self.runtime.stage,config,self.runtime.scm_enabled,self.runtime.scm_settings)
        self.terrain_view = SCMTerrainView(self.runtime.stage,self.runtime.terrain) if self.runtime.scm_enabled else None
        self.track_view = TrackVisuals(self.runtime.stage,self.runtime.track_motion) if self.runtime.track_motion else None
        self.follow_robot = False
        self.saved_route = False
        self.turn_running = False
        self.turn_cancel = False
        self.turn_status = ''
        self.export_status = ''
        self.suppress_manual = False
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
                    self.metrics_label = ui.Label('Slip / sinkage / RTF: waiting for Play', height=225, word_wrap=True)
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
                    with ui.CollapsableFrame('Turning experiments / comparison plots', collapsed=False, height=0):
                        with ui.VStack(spacing=4):
                            ui.Label('Fresh soil per trial | one CCW revolution', height=22)
                            with ui.HStack(height=26, spacing=5):
                                ui.Button('Spin on current surface', clicked_fn=lambda:self.start_turns(('spin',)))
                                ui.Button('Orbit on current surface', clicked_fn=lambda:self.start_turns(('orbit',)))
                            ui.Button('Compare both turns on all five surfaces', height=28,
                                      clicked_fn=lambda:self.start_turns(('spin','orbit'), all_surfaces=True))
                            ui.Button('Stop turning experiment', height=24, clicked_fn=self.pause)
                    ui.Button('Reset vehicle / keep ruts',height=26,
                              clicked_fn=lambda:asyncio.ensure_future(self.reset(clear_soil=False)))
                    ui.Button('Export telemetry CSV + plots',height=26,clicked_fn=self.export)
                    self.status_label = ui.Label('Ready on tile 1. Press Forward, then Play.',height=115,word_wrap=True)
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
        if self.suppress_manual:
            return
        if self.turn_running:
            self.turn_cancel = True
            self.runtime.turn_trial = None
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

    async def reset(self, clear_soil=True, scm_enabled=None, position=None, _turn_reset=False):
        if not _turn_reset:
            self.turn_cancel = True
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
        self.suppress_manual = True
        try:
            self.left.set_value(0.)
            self.right.set_value(0.)
        finally:
            self.suppress_manual = False
        await app.next_update_async()

    async def run_route(self):
        await self.reset()
        self.runtime.begin_route()
        self.runtime.timeline.play()

    def start_turns(self, modes, all_surfaces=False):
        if self.turn_running:
            return
        tiles = self.config['surfaces']
        if not all_surfaces:
            row = self.runtime.latest
            tile = self.runtime.environment.surface_at(row['x_m'], row['y_m']) if row else tiles[0]
            if tile is None:
                self.turn_status = 'Vehicle is outside all surfaces; reset first.'
                return
            tiles = [tile]
        self.turn_task = asyncio.ensure_future(self.run_turns(modes=modes, tiles=tiles))

    async def run_turns(self, modes=('spin','orbit'), tiles=None, settings=None):
        from turning_session import run_turning_batch
        return await run_turning_batch(self, modes=modes, tiles=tiles, settings=settings)

    def export(self):
        from ground_export import save_ground_run, plot_ground_run
        if not self.runtime.history:
            return None
        output = self.project/'results/five_surfaces'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        save_ground_run(self.runtime, output)
        self.last_export = output
        self.export_status = 'Generating line charts: '+str(output)

        async def finish_plots():
            summary = await plot_ground_run(output)
            self.export_status = ('Charts saved: ' if summary['plots_state'] == 'ready' else
                                  'CSV saved; chart generation failed: ')+str(output)
            print('GROUND_EXPORT '+self.export_status, flush=True)
        self.plot_task = asyncio.ensure_future(finish_plots())
        return output

    def on_update(self, event):
        if omni.usd.get_context().get_stage() != self.runtime.stage:
            return
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
                f"SCM: {row['scm_step_ms']:.2f} ms | mesh: {self.terrain_view.last_update_ms:.2f} ms\n"
                f"Pressure mean / max: {row['soil_pressure_mean_pa']/1000:.2f} / {row['soil_pressure_max_pa']/1000:.2f} kPa\n"
                f"Shear mean / max: {row['soil_shear_stress_mean_pa']/1000:.2f} / {row['soil_shear_stress_max_pa']/1000:.2f} kPa")
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
        if self.turn_status:
            self.status_label.text += '\n'+self.turn_status
        if self.export_status:
            self.status_label.text += '\n'+self.export_status
        if self.runtime.route_reason and not self.saved_route:
            self.export()
            self.saved_route=True
