"""--exec startup for the own crossing test, without the manual actuator panel."""

import asyncio
import json
from pathlib import Path
import sys
import traceback

import omni.kit.app

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "scripts"))


async def start():
    try:
        request_file = project / ".runtime/crossing_pending.json"
        request = json.loads(request_file.read_text())
        request_file.unlink()
        if request.get("kind") != "reference_physical_crossing":
            raise RuntimeError("Unsupported experiment request")
        from crossing_experiment import CrossingExperiment
        global _CROSSING_EXPERIMENT
        _CROSSING_EXPERIMENT = CrossingExperiment(project, request)
        await _CROSSING_EXPERIMENT.run()
    except Exception:
        target = project / ".runtime/crossing_startup_error.txt"
        target.write_text(traceback.format_exc(), encoding="utf-8")
        print("L17_CROSSING_STARTUP_ERROR\n" + traceback.format_exc(), flush=True)
        omni.kit.app.get_app().post_quit(3)


_CROSSING_TASK = asyncio.ensure_future(start())
