"""Own GUI startup: open existing scene, attach RGB-D, validate and keep it open."""
import asyncio
import json
from pathlib import Path
import sys
import traceback

import omni.kit.app
import omni.usd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))


async def main():
    try:
        config=json.loads((PROJECT/'configs/task.json').read_text())
        scene=(PROJECT/config['scene_file']).resolve()
        if not scene.is_relative_to(PROJECT):raise RuntimeError('Scene must remain in own project')
        success,error=await omni.usd.get_context().open_stage_async(str(scene))
        if not success:raise RuntimeError(error)
        for _ in range(8):await omni.kit.app.get_app().next_update_async()
        from robot_task_ui import show_robot_tools
        panel=show_robot_tools(PROJECT)
        from rgbd_sensor_rig import get_rig
        result=await get_rig(PROJECT).capture_all(run_calibration=True,include_reference=True)
        panel.label.text='RGB-D and calibration saved: '+Path(result['directory']).name
        panel.camera('Crossing')
        error_file=PROJECT/'results/rgbd/startup_error.json'
        if error_file.exists():
            backup=PROJECT/'.runtime/rgbd_backups/previous_startup_error.json'
            backup.write_bytes(error_file.read_bytes());error_file.unlink()
        print('L17_RGBD_CAPTURE_READY: '+result['directory'],flush=True)
    except Exception:
        report={'error':traceback.format_exc()}
        directory=PROJECT/'results/rgbd';directory.mkdir(parents=True,exist_ok=True)
        (directory/'startup_error.json').write_text(json.dumps(report,indent=2)+'\n')
        print(report['error'],flush=True)


asyncio.ensure_future(main())
