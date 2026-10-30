"""Load the five-surface scene and its local controls."""
import asyncio
import json
from pathlib import Path
import sys
import traceback

import omni.kit.app
import omni.usd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))


async def open_scene(scm_enabled=None):
    config=json.loads((PROJECT/'configs/five_surfaces.json').read_text())
    scene=(PROJECT/config['scene_file']).resolve()
    if not scene.is_relative_to(PROJECT):
        raise RuntimeError('Scene must be inside this project')
    success,error=await omni.usd.get_context().open_stage_async(str(scene))
    if not success:
        raise RuntimeError(str(error))
    app=omni.kit.app.get_app()
    for _ in range(5):
        await app.next_update_async()
    from ground_tools import GroundTools
    global GROUND_TOOLS
    GROUND_TOOLS=GroundTools(PROJECT,config,scm_enabled)
    print('FIVE_SURFACES_READY: '+str(scene),flush=True)
    return GROUND_TOOLS


async def start():
    try:
        await open_scene()
    except Exception:
        message=traceback.format_exc()
        (PROJECT/'.runtime/startup_error.txt').write_text(message)
        print(message,flush=True)


GROUND_TOOLS=None
if __name__ == '__main__':
    START_TASK=asyncio.ensure_future(start())
