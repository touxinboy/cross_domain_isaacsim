"""Renderer-free host for the same GroundRuntime and turning session as the GUI."""
import asyncio
import json
from pathlib import Path

import omni.kit.app
import omni.usd
from pxr import Gf, Usd

from ground_runtime import GroundRuntime
from scm_scene import configure_ground


class HeadlessTurningTools:
    def __init__(self, project, config):
        self.project, self.config = Path(project), config
        self.runtime = GroundRuntime(project, config, scm_enabled=True)
        configure_ground(self.runtime.stage, config, True, self.runtime.scm_settings)
        self.turn_running, self.turn_cancel, self.turn_status = False, False, ''

    async def reset(self, scm_enabled=True, position=None, _turn_reset=True):
        self.runtime.zero()
        self.runtime.timeline.stop()
        self.runtime.shutdown()
        await omni.kit.app.get_app().next_update_async()
        stage = omni.usd.get_context().get_stage()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            robot = stage.GetPrimAtPath('/World/Robot')
            robot.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*position))
            robot.GetAttribute('xformOp:orient').Set(Gf.Quatf(1))
        self.runtime = GroundRuntime(self.project, self.config, scm_enabled=True)
        configure_ground(stage, self.config, True, self.runtime.scm_settings)
        await omni.kit.app.get_app().next_update_async()

    async def run_turns(self, **kwargs):
        from turning_session import run_turning_batch
        return await run_turning_batch(self, **kwargs)


async def open_headless(project):
    config = json.loads((project/'configs/five_surfaces.json').read_text())
    success, error = await omni.usd.get_context().open_stage_async(str(project/config['scene_file']))
    if not success:
        raise RuntimeError(str(error))
    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()
    return HeadlessTurningTools(project, config)
