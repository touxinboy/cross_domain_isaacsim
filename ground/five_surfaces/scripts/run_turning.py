"""Native batch entry point for launch.py --exec-script scripts/run_turning.py."""
import asyncio
import faulthandler
import json
import os
from pathlib import Path
import sys
import traceback

import omni.kit.app
import omni.usd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'scripts'))


async def main():
    tools = None
    passed = False
    diagnostic = PROJECT/'.runtime/turning_startup.txt'
    diagnostic.parent.mkdir(exist_ok=True)
    handle = diagnostic.open('w', buffering=1)
    faulthandler.dump_traceback_later(90, repeat=True, file=handle)
    try:
        handle.write('Opening scene\n')
        print('TURNING_START opening scene', flush=True)
        if os.environ.get('GROUND_TURN_PHYSICS_ONLY') == '1':
            from turning_headless import open_headless
            tools = await asyncio.wait_for(open_headless(PROJECT), timeout=120.)
        else:
            from open_stage import open_scene
            tools = await asyncio.wait_for(open_scene(scm_enabled=True), timeout=120.)
        handle.write('Scene and controls ready\n')
        settings = json.loads((PROJECT/'configs/turning.json').read_text())
        # Optional small native smoke run; defaults always execute complete circles.
        if os.environ.get('GROUND_TURN_REVOLUTIONS'):
            settings['revolutions'] = float(os.environ['GROUND_TURN_REVOLUTIONS'])
        ids = os.environ.get('GROUND_TURN_SURFACES', '').split(',')
        tiles = [t for t in tools.config['surfaces'] if not ids[0] or t['id'] in ids]
        if not tiles:
            raise ValueError('No requested turning surfaces found')
        modes = tuple(os.environ.get('GROUND_TURN_MODES', 'spin,orbit').split(','))
        output = await tools.run_turns(modes=modes, tiles=tiles, settings=settings)
        manifest = json.loads((output/'experiments.json').read_text())
        passed = manifest['complete'] and not manifest.get('plot_error')
        print('TURNING_RESULTS '+str(output), flush=True)
    except Exception:
        print(traceback.format_exc(), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        handle.close()
        if tools:
            # Unsubscribe GUI updates before closing their USD stage.
            if hasattr(tools, 'update_sub'):
                tools.update_sub = None
            tools.runtime.zero()
            tools.runtime.timeline.stop()
            tools.runtime.shutdown()
        await omni.usd.get_context().close_stage_async()
        omni.kit.app.get_app().post_quit(0 if passed else 1)


TURNING_TASK = asyncio.ensure_future(main())
