"""Run in the own Isaac GUI: real static RGB-D + native runtime calibration.

No robot pose/velocity is set. Only the own session's runtime and sensor rig
are refreshed; all output remains under this project.
"""
import asyncio
import json
from pathlib import Path
import sys
import traceback
import uuid

import omni.usd
import omni.timeline

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"scripts"))


async def main(project=PROJECT):
    from navigation_io import own_path,read_json,write_json
    from pxr import UsdGeom
    output=project/"results/deployment/isaac_sensor_check.json"
    report={"passed":False,"validation_scope":"real_Isaac_static_RGBD_and_runtime_clock_binding",
            "robot_pose_or_velocity_overwrite":False}
    runtime=None
    try:
        stage=omni.usd.get_context().get_stage()
        expected=own_path(project,read_json(project/"configs/task.json")["scene_file"])
        if not stage or Path(stage.GetRootLayer().realPath).resolve()!=expected:
            raise RuntimeError("Open the configured own task scene before the sensor check")
        if UsdGeom.GetStageMetersPerUnit(stage)!=1 or str(UsdGeom.GetStageUpAxis(stage))!="Z":
            raise RuntimeError("The own scene must use metres and Z-up")
        panel=sys.modules.get("robot_task_ui")
        panel=getattr(panel,"_TOOLS",None)
        if panel and panel.busy():raise RuntimeError("An own task is already running")
        import robot_runtime
        existing=robot_runtime.get_robot_runtime()
        if existing and any(abs(x)>1e-8 for x in existing.commands["tracks"]+existing.commands["thrusters"]):
            raise RuntimeError("Zero the own manual commands before the sensor check")
        omni.timeline.get_timeline_interface().pause()
        robot_runtime.detach_robot_runtime()
        runtime=robot_runtime.attach_robot_runtime(project,"river")
        if not hasattr(runtime,"clock_epoch"):runtime.clock_epoch=uuid.uuid4().hex
        import rgbd_sensor_rig
        rgbd_sensor_rig.dispose_rig()
        source=project/"scripts/rgbd_sensor_rig.py"
        exec(compile(source.read_text(),str(source),"exec"),rgbd_sensor_rig.__dict__)
        summary=await rgbd_sensor_rig.get_rig(project).capture_all(run_calibration=True,runtime=runtime)
        from depth_localization import DepthFrame
        bound=[]
        for entry in summary["frame_files"]:
            frame=DepthFrame.load(own_path(project,entry["frame_file"]))
            if frame.metadata["controller_epoch"]!=runtime.clock_epoch:raise RuntimeError("Captured runtime epoch mismatch")
            if frame.metadata.get("render_physics_position_delta_m",1)>.05:raise RuntimeError("Render/native pose mismatch")
            bound.append(frame.metadata["frame_id"])
        report.update(passed=bool(summary["calibration"]["passed"] and len(bound)==3),
                      calibration=summary["calibration"],frame_ids=bound,capture_directory=summary["directory"])
    except Exception:report["error"]=traceback.format_exc()
    finally:
        if runtime:
            runtime.zero()
            if runtime.stage==omni.usd.get_context().get_stage():omni.timeline.get_timeline_interface().pause()
        write_json(output,report)
        print("L17_ISAAC_SENSOR_CHECK",json.dumps({"passed":report["passed"],"report":str(output)}),flush=True)


asyncio.ensure_future(main())
