"""Run once in the existing own Isaac Script Editor after uploading the update."""
import asyncio
import importlib
from pathlib import Path
import sys

import omni.usd
import omni.timeline

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"scripts"))


async def main(project=PROJECT):
    # Bind project now: Kit's exec/Script Editor may reuse its global namespace
    # for a later script while this coroutine is waiting for an app update.
    import json,time
    request=json.loads((project/".runtime/navigation/current_request.json").read_text())
    if float(request.get("start_deadline_wall_unix_s",0))<time.time():raise RuntimeError("Prepared task expired; prepare a new request")
    stage=omni.usd.get_context().get_stage()
    if not stage or not Path(stage.GetRootLayer().realPath).resolve().is_relative_to(project):
        raise RuntimeError("Open the own L17 robot scene; this script will not replace someone else's stage")
    task=json.loads((project/"configs/task.json").read_text())
    if Path(stage.GetRootLayer().realPath).resolve()!=(project/task["scene_file"]).resolve():
        raise RuntimeError("Open the configured own task scene first: "+task["scene_file"])
    import robot_task_ui
    panel=getattr(robot_task_ui,"_TOOLS",None)
    if panel and panel.busy():raise RuntimeError("An own task is still running")
    names=["navigation_io","navigation_contracts","depth_localization","amphibian_dynamics",
           "river_geometry","crossing_control","robot_runtime","navigation_observations",
           "approach_planning","navigation_execution","navigation_loop","navigation_experiment",
           "rgbd_sensor_rig","robot_task_ui"]
    for name in names:
        module=sys.modules.get(name)
        source=project/"scripts"/(name+".py")
        if module is not None and Path(module.__file__).resolve()!=source.resolve():
            raise RuntimeError("Another project owns module "+name+"; use a fresh own Kit process")
        compile(source.read_text(),str(source),"exec")
    import robot_runtime
    runtime=robot_runtime.get_robot_runtime()
    if runtime and any(abs(x)>1e-8 for x in runtime.commands["tracks"]+runtime.commands["thrusters"]):
        raise RuntimeError("Zero the own manual actuator commands first")
    omni.timeline.get_timeline_interface().pause()
    rig=sys.modules.get("rgbd_sensor_rig")
    if rig is not None:rig.dispose_rig()
    robot_runtime.detach_robot_runtime()
    if panel:panel.loop.cancel();panel.window.visible=False
    importlib.invalidate_caches()
    for name in names:
        module=importlib.import_module(name)
        source=project/"scripts"/(name+".py")
        exec(compile(source.read_text(),str(source),"exec"),module.__dict__)
    panel=sys.modules["robot_task_ui"].show_robot_tools(project)
    panel.start_navigation(request)


asyncio.ensure_future(main())
