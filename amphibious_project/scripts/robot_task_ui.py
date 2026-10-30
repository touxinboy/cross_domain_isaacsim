"""Manual actuator controls, component validation, and the own robot scene interface."""

import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
import re
import traceback

import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from pxr import Gf, Usd, UsdGeom

from river_geometry import RiverGeometry
from robot_assets import project_path, set_camera
from robot_runtime import attach_robot_runtime, get_robot_runtime
from robot_validation import RobotValidation

_TOOLS=None


class RobotTaskUI:
    def __init__(self,project):
        self.project=Path(project).resolve(strict=True)
        self.cfg=json.loads((self.project/"configs/robot_v1.json").read_text())
        self.scene_cfg=json.loads((self.project/"configs/river_forest.json").read_text())
        self.field=RiverGeometry(self.scene_cfg)
        self.has_jacket=bool(self.stage().GetPrimAtPath("/World/NavigationTargets/Jacket"))
        self.has_water_jacket=bool(self.stage().GetPrimAtPath("/World/NavigationTargets/WaterJacket"))
        if self.has_jacket:
            target_config="configs/jacket_targets_v2.json" if self.has_water_jacket else "configs/jacket_target_v1.json"
            target_data=json.loads((self.project/target_config).read_text())
            targets=target_data.get("targets",[target_data])
            for target in targets:
                self.scene_cfg["missions"].append({k:target[k] for k in
                    ["id","label","start_xy_m","goal_xy_m","goal_radius_m"]})
            self.scene_cfg["default_mission"]=target_data.get("default_target",targets[0]["id"])
        self.requests=project_path(project,".runtime/robot_ui");self.requests.mkdir(exist_ok=True)
        self.results=project_path(project,"results/robot_v1");self.results.mkdir(exist_ok=True)
        self.future=None;self.validator=None;self.mode="HYBRID";self.navigation=None
        self.left_track=ui.SimpleFloatModel(0);self.right_track=ui.SimpleFloatModel(0)
        self.left_thruster=ui.SimpleFloatModel(0);self.right_thruster=ui.SimpleFloatModel(0)
        self.mission_index=next(i for i,m in enumerate(self.scene_cfg["missions"]) if m["id"]==self.scene_cfg["default_mission"])
        self.water_model=ui.SimpleBoolModel(True)
        self.window=ui.Window("L17 Amphibious Robot",width=340,height=570)
        with self.window.frame:
            with ui.VStack(spacing=5,margin=9):
                ui.Label("AMPHIBIOUS ROBOT v1",style={"font_size":17})
                ui.Label("48 kg prototype | component tests",height=20)
                with ui.HStack(height=25,spacing=4):
                    ui.Button("Robot",clicked_fn=self.robot_view)
                    ui.Button("Overview",clicked_fn=lambda:self.camera("Overview"))
                    ui.Button("Riverbed",clicked_fn=self.bed_view)
                if self.has_jacket:
                    with ui.HStack(height=25,spacing=4):
                        ui.Button("Target",clicked_fn=lambda:self.camera("Target"))
                        if self.has_water_jacket:
                            ui.Button("Water target",clicked_fn=lambda:self.camera("WaterTarget"))
                        ui.Button("Crossing",clicked_fn=lambda:self.camera("Crossing"))
                with ui.HStack(height=20):
                    ui.CheckBox(model=self.water_model,width=24)
                    ui.Label("Show water surface")
                self.water_model.add_value_changed_fn(lambda m:self.water(m.as_bool))
                ui.Separator(height=5)
                combo=ui.ComboBox(self.mission_index,*[m["label"] for m in self.scene_cfg["missions"]],height=24)
                self.mission_combo=combo
                combo.model.add_item_changed_fn(lambda m,i:self.choose_mission(m.get_item_value_model().as_int))
                ui.Button("Reset to mission start",height=25,clicked_fn=self.reset_robot)
                ui.Label("Actuator mode",height=18)
                mode=ui.ComboBox(2,"LAND","WATER","HYBRID",height=23)
                mode.model.add_item_changed_fn(lambda m,i:self.set_mode(["LAND","WATER","HYBRID"][m.get_item_value_model().as_int]))
                for label,model in [("Left track (m/s)",self.left_track),("Right track (m/s)",self.right_track),
                                    ("Left thruster (-1..1)",self.left_thruster),("Right thruster (-1..1)",self.right_thruster)]:
                    with ui.HStack(height=21):
                        ui.Label(label,width=165)
                        ui.FloatSlider(model=model,min=-1,max=1)
                    model.add_value_changed_fn(lambda m:self.apply_controls())
                with ui.HStack(height=26,spacing=4):
                    ui.Button("Play",clicked_fn=self.play)
                    ui.Button("Zero",clicked_fn=self.zero)
                    ui.Button("Stop",clicked_fn=self.stop)
                ui.Separator(height=5)
                ui.Button("Run land / float / thrust checks",height=30,clicked_fn=self.start_suite)
                ui.Button("Export current telemetry CSV",height=24,clicked_fn=self.export_csv)
                self.label=ui.Label("Robot ready. Navigation is not installed.",height=42,word_wrap=True)
        stage_window=ui.Workspace.get_window("Stage")
        if stage_window:self.window.dock_in(stage_window,ui.DockPosition.BOTTOM,.65)
        self.restore_river()
        if self.has_jacket:self.camera("Crossing")
        self.loop=asyncio.ensure_future(self.request_loop())

    def busy(self):
        return bool((self.future and not self.future.done()) or
                    (self.navigation and self.navigation.future and not self.navigation.future.done()))

    def start_navigation(self,request):
        if self.busy():raise RuntimeError("An own validation or navigation task is already running")
        if request.get('schema')=='L17.semantic_navigation_run.v1':
            from semantic_navigation_experiment import SemanticNavigationExperiment
            self.navigation=SemanticNavigationExperiment(self.project,request)
        else:
            from navigation_experiment import NavigationExperiment
            self.navigation=NavigationExperiment(self.project,request)
        self.label.text="Navigation interface: "+request["mode"]+". Stop ends this task."
        result=self.navigation.start()
        def completed(future):
            error="cancelled" if future.cancelled() else str(future.exception() or "")
            self.label.text="Navigation: "+(error or self.navigation.reason or "finished")+". Logs: "+self.navigation.output.name
        self.navigation.future.add_done_callback(completed)
        return result

    def runtime(self):
        return get_robot_runtime()

    def stage(self):
        stage=omni.usd.get_context().get_stage()
        if not stage or not Path(stage.GetRootLayer().realPath).resolve().is_relative_to(self.project):
            raise RuntimeError("Robot controls only operate on this project.")
        return stage

    def set_mode(self,mode):
        if self.busy():return
        self.mode=mode;self.apply_controls()

    def apply_controls(self):
        if self.busy():return
        r=self.runtime()
        if r:r.set_commands(self.mode,[self.left_track.as_float,self.right_track.as_float],
                             [self.left_thruster.as_float,self.right_thruster.as_float])

    def zero(self):
        if self.busy():return
        for model in [self.left_track,self.right_track,self.left_thruster,self.right_thruster]:model.set_value(0)
        if self.runtime():self.runtime().zero()

    def play(self):
        if self.busy():return
        omni.timeline.get_timeline_interface().play()

    def stop(self):
        if self.navigation and self.navigation.future and not self.navigation.future.done():
            self.navigation.request_stop();return
        if self.busy():return
        self.zero();omni.timeline.get_timeline_interface().stop()

    def choose_mission(self,index):
        if self.busy():return
        self.mission_index=index;self.reset_robot()

    def reset_robot(self):
        if self.busy():return
        self.stop()
        stage=self.stage();mission=self.scene_cfg["missions"][self.mission_index]
        x,y=mission["start_xy_m"];z=self.field.mesh_height(x,y)+.24
        # PhysX writes poses to the root layer. A stronger session pose freezes rendering.
        with Usd.EditContext(stage,stage.GetRootLayer()):
            robot=stage.GetPrimAtPath("/World/Robot")
            robot.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x,y,z))
            robot.GetAttribute("xformOp:orient").Set(Gf.Quatf(1))
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for name,key in [("Start","start_xy_m"),("Goal","goal_xy_m")]:
                a,b=mission[key]
                stage.GetPrimAtPath("/World/TaskMarkers/"+name).GetAttribute("xformOp:translate").Set(Gf.Vec3d(a,b,self.field.mesh_height(a,b)))
        self.robot_view()

    def camera(self,name):
        stage=self.stage()
        if name not in ["Robot","Overview","Crossing","Riverbed","Target","WaterTarget"]:raise ValueError("Unknown own camera")
        path="/World/Cameras/"+name
        if not stage.GetPrimAtPath(path):raise RuntimeError("Camera not available in this test scene")
        get_active_viewport().set_active_camera(path)

    def robot_view(self):
        stage=self.stage();r=self.runtime()
        if r and r.latest and omni.timeline.get_timeline_interface().is_playing():
            p=[r.latest[k] for k in ["x_m","y_m","z_m"]]
        else:
            p=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Robot")).ExtractTranslation()
        set_camera(stage,"/World/Cameras/Robot",(p[0]-1.6,p[1]-1.9,p[2]+1.05),(p[0],p[1],p[2]+.06),35)
        self.camera("Robot")

    def water(self,visible):
        if self.busy():return
        stage=self.stage();path="/World/Water/Surface" if stage.GetPrimAtPath("/World/Water/Surface") else "/World/Water"
        prim=stage.GetPrimAtPath(path)
        if prim:
            with Usd.EditContext(stage,stage.GetSessionLayer()):
                obj=UsdGeom.Imageable(prim);obj.MakeVisible() if visible else obj.MakeInvisible()

    def bed_view(self):
        if self.busy():return
        self.water_model.set_value(False);self.camera("Riverbed")

    def restore_river(self):
        r=self.runtime()
        if not r or r.stage!=omni.usd.get_context().get_stage():attach_robot_runtime(self.project,"river")
        stage=self.stage()
        start=stage.GetPrimAtPath("/World/TaskMarkers/Start")
        if start:
            with Usd.EditContext(stage,stage.GetSessionLayer()):
                UsdGeom.Imageable(start).MakeInvisible()
        self.robot_view()
        import carb
        settings=carb.settings.get_settings()
        settings.set_bool("/app/viewport/grid/enabled",False)
        settings.set_bool("/app/viewport/show/lights",False)
        settings.set_bool("/app/viewport/show/camera",False)
        settings.set_int("/persistent/app/viewport/displayOptions",1<<10)
        if self.validator and self.validator.summary.get("passed"):
            self.label.text="Land, float and thrust checks passed. Manual controls are ready."

    def start_suite(self):
        if self.busy():return {"running":True,"run_directory":str(self.validator.output) if self.validator else None}
        self.stop()
        asset=project_path(self.project,"robots/tracked_twin_thruster_v1.usda")
        self.validator=RobotValidation(self.project,asset,self.cfg,self)
        self.label.text="Running component checks. Measurements are saved automatically."
        self.future=asyncio.ensure_future(self.validator.run())
        return {"running":True,"run_directory":str(self.validator.output)}

    def export_csv(self):
        if self.busy():return
        r=self.runtime()
        if r and r.history:
            file=self.results/("manual_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".csv")
            r.save_csv(file);self.label.text="Saved telemetry: "+file.name

    def status(self):
        return {"scene":self.stage().GetRootLayer().realPath,"validation_running":self.busy(),
                "navigation":self.navigation.status() if self.navigation else None,
                "validation_directory":str(self.validator.output) if self.validator else None,
                "runtime":self.runtime().status() if self.runtime() else None}

    async def capture(self,name):
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.png",name):raise ValueError("Use a simple PNG filename")
        for _ in range(35):await omni.kit.app.get_app().next_update_async()
        output=self.results/name
        job=capture_viewport_to_file(get_active_viewport(),str(output));await job.wait_for_result()
        return str(output)

    async def request_loop(self):
        while self.window:
            for file in sorted(self.requests.glob("*.request.json")):
                try:
                    data=json.loads(file.read_text());action=data.get("action")
                    if action=="status":result=self.status()
                    elif action=="navigation_start":result=self.start_navigation(data["request"])
                    elif action=="navigation_stop":
                        if self.navigation:self.navigation.request_stop()
                        result=self.status()
                    elif action=="suite":result=self.start_suite()
                    elif action=="camera":self.camera(data["name"]);result=self.status()
                    elif action=="robot_view":self.robot_view();result=self.status()
                    elif action=="capture":result={"file":await self.capture(data["filename"])}
                    elif action=="rgbd_capture":
                        if self.busy():raise RuntimeError("Component validation is running")
                        import rgbd_sensor_rig
                        if data.get("reload",False):
                            try:rgbd_sensor_rig.dispose_rig()
                            finally:
                                # Kit's accelerated importer can reuse cached
                                # source on importlib.reload. Reload only this
                                # fixed, own-project module from its actual file.
                                source=self.project/"scripts/rgbd_sensor_rig.py"
                                exec(compile(source.read_text(),str(source),"exec"),rgbd_sensor_rig.__dict__)
                        result=await rgbd_sensor_rig.get_rig(self.project).capture_all(
                            run_calibration=data.get("calibration",False),include_reference=data.get("include_reference",False))
                    elif action=="commands":
                        if self.busy():raise RuntimeError("Component validation is running")
                        self.runtime().set_commands(data.get("mode","HYBRID"),data.get("tracks"),data.get("thrusters"))
                        result=self.status()
                    elif action=="play":self.play();result=self.status()
                    elif action=="stop":self.stop();result=self.status()
                    elif action=="save":
                        if self.busy():raise RuntimeError("Do not save a transient validation scene")
                        self.stage().GetRootLayer().Save();result=self.status()
                    else:raise ValueError("Unsupported robot command")
                    response={"ok":True,"result":result}
                except Exception:response={"ok":False,"error":traceback.format_exc()}
                file.with_name(file.name.replace(".request.json",".result.json")).write_text(json.dumps(response,indent=2)+"\n")
                file.unlink()
            await asyncio.sleep(.2)


def show_robot_tools(project):
    global _TOOLS
    if _TOOLS:
        _TOOLS.loop.cancel();_TOOLS.window.visible=False
    _TOOLS=RobotTaskUI(project)
    return _TOOLS
