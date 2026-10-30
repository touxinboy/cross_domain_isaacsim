"""Create the robot, add it to an own scene, and run the component experiments."""

import asyncio
import hashlib
import json
from pathlib import Path
import sys
import traceback

import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"scripts"))
from river_geometry import RiverGeometry
from robot_assets import build_robot_asset, add_robot, set_physics, set_camera, project_path
from robot_task_ui import show_robot_tools


async def main():
    cfg=json.loads((PROJECT/"configs/robot_v1.json").read_text())
    base=project_path(PROJECT,"scenes/river_forest_v1.usda")
    original_hash=hashlib.sha256(base.read_bytes()).hexdigest()
    asset=build_robot_asset(PROJECT,cfg)
    file=project_path(PROJECT,"scenes/river_robot_v1.usda")
    if not file.exists():
        stage=Usd.Stage.CreateNew(str(file))
        stage.GetRootLayer().subLayerPaths=["river_forest_v1.usda"]
        UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,"Z")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        set_physics(stage,cfg)
        scene_cfg=json.loads((PROJECT/"configs/river_forest.json").read_text())
        field=RiverGeometry(scene_cfg)
        mission=next(m for m in scene_cfg["missions"] if m["id"]==scene_cfg["default_mission"])
        x,y=mission["start_xy_m"];z=field.mesh_height(x,y)+.24
        add_robot(stage,PROJECT,asset,(x,y,z))
        set_camera(stage,"/World/Cameras/Robot",(x-1.6,y-1.9,z+1.05),(x,y,z+.06),35)
        water=stage.GetPrimAtPath("/World/Water/Surface")
        water.SetCustomDataByKey("water_dynamics_enabled",True)
        water.SetCustomDataByKey("purpose_note","Visual surface; distributed buoyancy and drag run in robot_runtime.py.")
        stage.GetRootLayer().customLayerData={"creator":"L17","robot_runtime_required":True}
        stage.GetRootLayer().Save()
    saved=Usd.Stage.Open(str(file))
    saved.SetStartTimeCode(0);saved.SetEndTimeCode(36000);saved.GetRootLayer().Save()
    ok,error=await omni.usd.get_context().open_stage_async(str(file))
    if not ok:raise RuntimeError(error)
    taskfile=project_path(PROJECT,"configs/task.json")
    task=json.loads(taskfile.read_text())
    task.update({"task_name":"river_robot_v1","scene_file":"scenes/river_robot_v1.usda","robot_config":"configs/robot_v1.json",
                 "description":"Force-driven tracked amphibian with buoyancy, drag and two thrusters. Navigation and learned switching are not implemented."})
    taskfile.write_text(json.dumps(task,indent=2)+"\n")
    tools=show_robot_tools(PROJECT)
    print("L17_ROBOT_SCENE_READY "+str(file),flush=True)
    tools.start_suite()
    await tools.future
    summary=tools.validator.summary
    if not summary.get("passed"):
        print("L17_ROBOT_COMPONENT_CHECKS_FAILED "+str(tools.validator.output),flush=True)
        return
    cfg["status"]="prototype_component_checks_passed_not_hardware_calibrated"
    (PROJECT/"configs/robot_v1.json").write_text(json.dumps(cfg,indent=2)+"\n")
    (PROJECT/"results/robot_v1/latest_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if hashlib.sha256(base.read_bytes()).hexdigest()!=original_hash:raise RuntimeError("Original forest scene changed")
    await tools.capture("robot_in_river_scene.png")
    omni.usd.get_context().get_stage().GetRootLayer().Save()
    print("L17_ROBOT_ALL_CHECKS_PASSED "+str(tools.validator.output),flush=True)


async def run():
    try:await main()
    except Exception:
        p=project_path(PROJECT,"results/robot_v1");p.mkdir(exist_ok=True)
        (p/"build_error.txt").write_text(traceback.format_exc())
        print("L17_ROBOT_BUILD_ERROR\n"+traceback.format_exc(),flush=True)


asyncio.ensure_future(run())
