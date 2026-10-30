"""Own-scene view controls and an L17-only file interface for scene inspection."""

import asyncio
import json
from pathlib import Path
import re
import traceback

import omni.kit.app
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from pxr import Gf, Usd, UsdGeom

from river_geometry import RiverGeometry

_TOOLS = None


class RiverSceneTools:
    def __init__(self, project):
        self.project = Path(project).resolve(strict=True)
        root = Path("/home/ubuntu/sdc/L17").resolve(strict=True)
        if not self.project.is_relative_to(root):
            raise RuntimeError("Scene controls must be inside L17.")
        self.config = json.loads((self.project / "configs/river_forest.json").read_text())
        self.field = RiverGeometry(self.config)
        self.mission_index = next(i for i, m in enumerate(self.config["missions"])
                                  if m["id"] == self.config["default_mission"])
        self.results = self.project / "results/river_forest_v1"
        self.results.mkdir(exist_ok=True)
        self.requests = self.project / ".runtime/river_ui"
        self.requests.mkdir(exist_ok=True)
        self.water_model = ui.SimpleBoolModel(True)
        self.water_model.add_value_changed_fn(lambda model: self.set_water(model.as_bool))
        self.window = ui.Window("L17 River Task", width=330, height=285)
        with self.window.frame:
            with ui.VStack(spacing=6, margin=10):
                ui.Label("FOREST RIVER", style={"font_size": 18})
                ui.Label("32 x 72 m  |  Water level: 0 m")
                with ui.HStack(height=26, spacing=4):
                    ui.Button("Overview", clicked_fn=lambda: self.set_camera("Overview"))
                    ui.Button("Crossing", clicked_fn=lambda: self.set_camera("Crossing"))
                    ui.Button("Riverbed", clicked_fn=self.bed_view)
                with ui.HStack(height=22):
                    ui.CheckBox(model=self.water_model, width=24)
                    ui.Label("Show water surface")
                ui.Separator(height=6)
                ui.Label("Mission setup")
                combo = ui.ComboBox(self.mission_index, *[m["label"] for m in self.config["missions"]], height=26)
                self.mission_combo = combo
                combo.model.add_item_changed_fn(lambda model, item: self.set_mission(model.get_item_value_model().as_int))
                self.info = ui.Label("", height=40, word_wrap=True)
                ui.Label("Green: start   |   Red: goal", height=20)
                ui.Label("Environment ready. Robot and control are next.", height=35, word_wrap=True)
        stage_window = ui.Workspace.get_window("Stage")
        if stage_window:
            self.window.dock_in(stage_window, ui.DockPosition.BOTTOM, 0.4)
        self.set_mission(self.mission_index)
        self.set_camera("Overview")
        self.hide_editor_overlays()
        self.loop_task = asyncio.ensure_future(self.request_loop())

    def own_stage(self):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise RuntimeError("No scene is open.")
        file = Path(stage.GetRootLayer().realPath).resolve()
        if not file.is_relative_to(self.project) or not file.name.startswith("river_forest"):
            raise RuntimeError("These controls only operate on this project's forest scene.")
        return stage

    def hide_editor_overlays(self):
        # Legacy settings remain supported by the installed Kit 106 viewport.
        # Only this isolated app's runtime settings are affected.
        import carb
        settings = carb.settings.get_settings()
        settings.set_bool("/app/viewport/grid/enabled", False)
        settings.set_bool("/app/viewport/show/lights", False)
        settings.set_bool("/app/viewport/show/camera", False)
        settings.set_int("/persistent/app/viewport/displayOptions", 1 << 10)
        omni.usd.get_context().get_selection().set_selected_prim_paths([], False)

    def set_camera(self, name):
        self.own_stage()
        if name not in ["Overview", "Crossing", "Riverbed"]:
            raise ValueError("Unknown scene camera.")
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera("/World/Cameras/" + name)

    def set_water(self, visible):
        stage = self.own_stage()
        # View switches are temporary and never rewrite the saved USD file.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            image = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Water/Surface"))
            image.MakeVisible() if visible else image.MakeInvisible()

    def bed_view(self):
        self.water_model.set_value(False)
        self.set_camera("Riverbed")

    def set_mission(self, index):
        if index < 0 or index >= len(self.config["missions"]):
            raise ValueError("Unknown mission.")
        stage = self.own_stage()
        mission = self.config["missions"][index]
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for name, key in [("Start", "start_xy_m"), ("Goal", "goal_xy_m")]:
                x, y = mission[key]
                p = stage.GetPrimAtPath("/World/TaskMarkers/" + name)
                p.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, self.field.mesh_height(x, y)))
        self.mission_index = index
        value_model = self.mission_combo.model.get_item_value_model()
        if value_model.as_int != index:
            value_model.set_value(index)
        self.info.text = f"Start: {mission['start_xy_m']}\nGoal:  {mission['goal_xy_m']}"

    def status(self):
        stage = self.own_stage()
        viewport = get_active_viewport()
        return {"scene": stage.GetRootLayer().realPath,
                "camera": str(viewport.get_active_camera()) if viewport else None,
                "mission": self.config["missions"][self.mission_index]["id"],
                "mission_combo_index": self.mission_combo.model.get_item_value_model().as_int,
                "water_visible": self.water_model.as_bool, "water_dynamics_enabled": False,
                "robot_present": False, "root_layer_dirty": stage.GetRootLayer().dirty}

    async def capture(self, filename):
        self.own_stage()
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.png", filename):
            raise ValueError("Screenshot name must be a simple PNG filename.")
        viewport = get_active_viewport()
        if not viewport:
            raise RuntimeError("No active viewport.")
        for _ in range(25):
            await omni.kit.app.get_app().next_update_async()
        job = capture_viewport_to_file(viewport, str(self.results / filename))
        await job.wait_for_result()
        return str(self.results / filename)

    async def capture_views(self):
        self.water_model.set_value(True)
        self.set_camera("Overview")
        await self.capture("overview.png")
        self.set_camera("Crossing")
        await self.capture("crossing.png")
        self.bed_view()
        await self.capture("riverbed.png")
        self.water_model.set_value(True)
        self.set_camera("Overview")

    async def request_loop(self):
        # No network listener and no arbitrary-code action: cameras, visibility and screenshots only.
        while self.window:
            for request in sorted(self.requests.glob("*.request.json")):
                try:
                    data = json.loads(request.read_text())
                    action = data.get("action")
                    if action == "status":
                        result = self.status()
                    elif action == "camera":
                        self.set_camera(data["name"])
                        result = self.status()
                    elif action == "water":
                        self.water_model.set_value(bool(data["visible"]))
                        result = self.status()
                    elif action == "mission":
                        self.set_mission(int(data["index"]))
                        result = self.status()
                    elif action == "capture":
                        result = {"file": await self.capture(data["filename"])}
                    elif action == "capture_views":
                        await self.capture_views()
                        result = self.status()
                    elif action == "save":
                        stage = self.own_stage()
                        if not stage.GetRootLayer().Save():
                            raise RuntimeError("Could not save the project's forest scene.")
                        result = self.status()
                    elif action == "reload":
                        stage = self.own_stage()
                        if stage.GetRootLayer().dirty:
                            raise RuntimeError("Save this scene before reloading it.")
                        path = stage.GetRootLayer().realPath
                        ok, error = await omni.usd.get_context().open_stage_async(path)
                        if not ok:
                            raise RuntimeError(error)
                        self.set_mission(self.mission_index)
                        self.set_water(self.water_model.as_bool)
                        self.set_camera("Overview")
                        result = self.status()
                    else:
                        raise ValueError("Unsupported scene-inspection action.")
                    response = {"ok": True, "result": result}
                except Exception:
                    response = {"ok": False, "error": traceback.format_exc()}
                output = request.with_name(request.name.replace(".request.json", ".result.json"))
                output.write_text(json.dumps(response, indent=2) + "\n")
                request.unlink()
            await asyncio.sleep(0.25)


def show_scene_tools(project):
    global _TOOLS
    if _TOOLS:
        _TOOLS.loop_task.cancel()
        _TOOLS.window.visible = False
    _TOOLS = RiverSceneTools(project)
    return _TOOLS
