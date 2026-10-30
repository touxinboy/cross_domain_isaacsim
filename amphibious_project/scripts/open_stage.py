"""Executed by the full Isaac Sim app after startup."""

import asyncio
import json
import sys
from pathlib import Path

import carb
import omni.kit.app
import omni.usd


async def open_project_scene():
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs/task.json").read_text())
    scene = (project / config.get("scene_file", "scenes/my_river.usda")).resolve()
    if not scene.is_relative_to(project.resolve()):
        raise RuntimeError("The startup scene must stay inside this project.")
    if not scene.is_file():
        carb.log_error(f"L17 scene is missing: {scene}")
        return
    success, error = await omni.usd.get_context().open_stage_async(str(scene))
    if not success:
        carb.log_error(f"L17 scene open failed: {error}")
        return
    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()
    if scene.name.startswith("river_robot"):
        sys.path.insert(0, str(project / "scripts"))
        from robot_task_ui import show_robot_tools
        show_robot_tools(project)
    elif scene.name.startswith("river_forest"):
        sys.path.insert(0, str(project / "scripts"))
        from river_task_ui import show_scene_tools
        show_scene_tools(project)
    carb.log_info(f"L17_OWN_SCENE_READY: {scene}")
    print(f"L17_OWN_SCENE_READY: {scene}", flush=True)


asyncio.ensure_future(open_project_scene())
