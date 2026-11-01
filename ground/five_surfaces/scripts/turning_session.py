"""Shared asynchronous native experiment orchestration; no viewport imports."""
import asyncio
from datetime import datetime
import json
import math
import time
import traceback

import omni.kit.app

from turning_experiments import TurningTrial, METRIC_DEFINITIONS


async def run_turning_batch(self, modes=('spin','orbit'), tiles=None, settings=None):
    if self.turn_running:
        raise RuntimeError('A turning experiment is already running')
    tiles = self.config['surfaces'] if tiles is None else tiles
    settings = json.loads((self.project/'configs/turning.json').read_text()) if settings is None else settings
    plans = [TurningTrial(mode, tile, settings, self.runtime.cfg['tracks']['center_separation_m'])
             for mode in modes for tile in tiles]
    if not plans:
        raise ValueError('At least one turning trial is required')
    output = self.project/'results/turning'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(engine='native CPU PhysX + SCM + circulating shoes', complete=False,
                    visualization='viewport' if hasattr(self, 'terrain_view') else 'none',
                    rtf_scope='Includes viewport and mesh updates' if hasattr(self, 'terrain_view') else 'No viewport rendering or terrain mesh updates',
                    settings=settings, scm_settings=self.runtime.scm_settings,
                    track_settings=self.runtime.track_settings,
                    metric_definitions=METRIC_DEFINITIONS, trials=[])
    self.turn_running, self.turn_cancel = True, False
    app = omni.kit.app.get_app()

    def save_manifest():
        (output/'experiments.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')

    save_manifest()
    try:
        for index, trial in enumerate(plans):
            if self.turn_cancel:
                break
            self.turn_status = f"Turning {index+1}/{len(plans)}: {trial.tile['id']} {trial.mode}"
            await self.reset(scm_enabled=True, position=trial.start_position, _turn_reset=True)
            runtime = self.runtime
            runtime.turn_trial = trial
            deadline = time.perf_counter()+settings['wall_timeout_s']
            runtime.timeline.play()
            while trial.reason is None:
                await app.next_update_async()
                if self.turn_cancel or self.runtime is not runtime:
                    trial.reason = 'cancelled'
                elif runtime.error:
                    trial.reason = 'physics_error'
                elif time.perf_counter() > deadline:
                    trial.reason = 'wall_timeout'
            if self.runtime is runtime:
                runtime.zero()
                runtime.timeline.pause()
                runtime.turn_trial = None
            stem = trial.tile['id']+'_'+trial.mode
            runtime.save_csv(output/(stem+'.csv'))
            runtime.terrain.save(output/(stem+'_soil.npz'))
            result = dict(surface=trial.tile['id'], mode=trial.mode, result=trial.reason,
                          csv=stem+'.csv', samples=len(runtime.history),
                          center_m=trial.center.tolist(), target_radius_m=trial.radius,
                          progress_rad=trial.progress, physics_error=runtime.error,
                          simulated_s=runtime.sim_time)
            if runtime.latest:
                for key in ('rtf_average', 'scm_step_ms', 'turn_radius_error_m'):
                    value = runtime.latest.get(key)
                    result[key] = value if value is not None and math.isfinite(value) else None
            manifest['trials'].append(result)
            save_manifest()
            print('TURNING_TRIAL '+json.dumps(result), flush=True)
            if trial.reason in ('cancelled', 'physics_error', 'wall_timeout'):
                break
        manifest['complete'] = (len(manifest['trials']) == len(plans) and
                                all(t['result'] == 'completed' for t in manifest['trials']))
        save_manifest()
        if any(t['samples'] for t in manifest['trials']):
            self.turn_status = 'Generating comparison plots...'
            try:
                from plot_turning import generate_plots
                manifest['plots'] = await asyncio.to_thread(generate_plots, output)
            except Exception:
                manifest['plot_error'] = traceback.format_exc()
                print(manifest['plot_error'], flush=True)
    except Exception:
        manifest['error'] = traceback.format_exc()
        self.runtime.zero()
        self.runtime.timeline.pause()
        print(manifest['error'], flush=True)
    finally:
        self.turn_running = False
        save_manifest()
        self.turn_status = ('Turning complete' if manifest['complete'] else 'Turning stopped / incomplete')+f': {output}'
        print('TURNING_OUTPUT '+str(output), flush=True)
    return output
