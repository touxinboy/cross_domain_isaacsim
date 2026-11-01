"""Native continuous traversal and line charts; launched by launch.py straight."""
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

import omni.kit.app
import omni.usd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'scripts'))


async def main():
    tools, output, passed = None, None, False
    try:
        if os.environ.get('GROUND_TURN_PHYSICS_ONLY') == '1':
            from turning_headless import open_headless
            tools = await asyncio.wait_for(open_headless(PROJECT), timeout=120.)
            await tools.reset(position=tools.config['start_position_m'])
            tools.runtime.begin_route()
            tools.runtime.timeline.play()
        else:
            from open_stage import open_scene
            tools = await asyncio.wait_for(open_scene(scm_enabled=True), timeout=120.)
            await tools.run_route()
        runtime = tools.runtime
        deadline = time.perf_counter()+240.
        print('STRAIGHT_START', flush=True)
        while runtime.running_route:
            await omni.kit.app.get_app().next_update_async()
            if runtime.error:
                raise RuntimeError(runtime.error)
            if time.perf_counter() > deadline:
                runtime.finish_route('wall_timeout')
        runtime.zero()
        runtime.timeline.pause()
        from ground_export import save_ground_run, plot_ground_run
        if hasattr(tools, 'update_sub'):
            # on_update already starts the automatic GUI export at route end.
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()
            if tools.saved_route:
                await tools.plot_task
                output = tools.last_export
            else:
                output = tools.export()
                await tools.plot_task
        else:
            output = PROJECT/'results/five_surfaces'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
            save_ground_run(runtime, output)
            await plot_ground_run(output)
        summary = json.loads((output/'summary.json').read_text())
        expected = {t['id'] for t in tools.config['surfaces']}
        passed = runtime.route_reason == 'completed' and runtime.visited == expected and summary['plots_state'] == 'ready'
        print('STRAIGHT_RESULTS '+str(output), flush=True)
        print('STRAIGHT_COMPLETE '+json.dumps(dict(passed=passed, result=runtime.route_reason,
              visited=sorted(runtime.visited), samples=len(runtime.history), plots=summary.get('plots', []))), flush=True)
    except Exception:
        print(traceback.format_exc(), flush=True)
        if tools and tools.runtime.history and output is None:
            from ground_export import save_ground_run, plot_ground_run
            tools.runtime.zero()
            tools.runtime.timeline.pause()
            output = PROJECT/'results/five_surfaces'/datetime.now().strftime('failed_%Y%m%d_%H%M%S_%f')
            save_ground_run(tools.runtime, output)
            await plot_ground_run(output)
    finally:
        if tools:
            if hasattr(tools, 'update_sub'):
                tools.update_sub = None
            tools.runtime.zero()
            tools.runtime.timeline.stop()
            tools.runtime.shutdown()
        await omni.usd.get_context().close_stage_async()
        omni.kit.app.get_app().post_quit(0 if passed else 1)


STRAIGHT_TASK = asyncio.ensure_future(main())
