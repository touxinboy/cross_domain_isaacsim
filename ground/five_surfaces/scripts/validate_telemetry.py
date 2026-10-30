"""Run in Isaac Sim: exercise the actual panel, native motion and CSV export."""
import asyncio
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import omni.kit.app
import omni.usd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))


async def validate():
    app = omni.kit.app.get_app()
    output = PROJECT / 'results/telemetry_validation'
    output.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'tested_project': str(PROJECT), 'checks': {}}
    tools = None

    async def advance(duration):
        target = tools.runtime.sim_time + duration
        deadline = time.perf_counter() + 120
        next_progress = time.perf_counter() + 5
        while tools.runtime.sim_time < target:
            if tools.runtime.error:
                raise RuntimeError(tools.runtime.error)
            if time.perf_counter() > deadline:
                raise RuntimeError('Physics advance timeout')
            if time.perf_counter() > next_progress:
                print('TELEMETRY_PROGRESS '+json.dumps({
                    'sim_s': tools.runtime.sim_time, 'target_s': target,
                    'timeline_s': tools.runtime.timeline.get_current_time(),
                    'playing': tools.runtime.timeline.is_playing(),
                    'initialization_attempts': tools.runtime.initialization_attempts,
                    'samples': len(tools.runtime.history)}), flush=True)
                next_progress = time.perf_counter() + 5
            await app.next_update_async()

    try:
        from open_stage import open_scene
        from ground_telemetry import metric_text
        tools = await open_scene(scm_enabled=False)
        checks = report['checks']
        tools.runtime.timeline.play()
        await advance(1.)
        checks['stationary_slip_is_unavailable'] = tools.runtime.latest['left_slip_valid'] == 0
        checks['initial_drop_not_sinkage'] = max(r['left_sinkage_m'] for r in tools.runtime.history) < .002
        tools.drive(.8, .8)
        await advance(1.)
        checks['moving_slip_is_available'] = tools.runtime.latest['left_slip_valid'] == 1
        checks['positive_startup_slip_recorded'] = any(
            r['left_slip_valid'] and r['left_slip_ratio'] > .1 for r in tools.runtime.history)
        tools.runtime.timeline.pause()
        await app.next_update_async()
        frozen_sim = tools.runtime.sim_time
        frozen_active = tools.runtime.performance.wall_s
        paused_at = time.perf_counter()
        while time.perf_counter() - paused_at < 1.:
            await app.next_update_async()
        checks['pause_freezes_sim_and_active_clock'] = (
            tools.runtime.sim_time == frozen_sim and tools.runtime.performance.wall_s == frozen_active)
        resumed_at = time.perf_counter()
        tools.runtime.timeline.play()
        await advance(.5)
        latest = tools.runtime.latest
        checks['resume_excludes_pause_duration'] = abs(
            (latest['active_wall_time_s'] - frozen_active) - (time.perf_counter() - resumed_at)) < .25
        tools.drive(-.5, -.5)
        await advance(2.)
        checks['reverse_motion_recorded'] = tools.runtime.latest['vx_m_s'] < -.1
        tools.drive(-.4, .4)
        await advance(1.)
        checks['turning_lateral_slip_recorded'] = tools.runtime.latest['left_lateral_slip_rms_m_s'] > .01
        tools.pause()
        await app.next_update_async()
        checks['panel_matches_latest_sample'] = tools.metrics_label.text == metric_text(tools.runtime.latest)
        manual = tools.export()
        report['manual_export'] = str(manual)
        checks['manual_export_contains_metric_definitions'] = 'telemetry' in json.loads((manual/'summary.json').read_text())
        await tools.reset()
        checks['reset_clears_history_clock_and_slip_distance'] = (
            not tools.runtime.history and tools.runtime.performance.wall_s == 0.
            and tools.runtime.telemetry.slip_distance == [0., 0.])
        checks['reset_clears_display'] = 'waiting for Play' in tools.metrics_label.text
        tools.runtime.begin_route()
        tools.runtime.timeline.play()
        deadline = time.perf_counter() + 300
        while tools.runtime.running_route:
            if tools.runtime.error:
                raise RuntimeError(tools.runtime.error)
            if time.perf_counter() > deadline:
                raise RuntimeError('Route timeout')
            await app.next_update_async()
        await app.next_update_async()
        runtime = tools.runtime
        checks['route_completed'] = runtime.route_reason == 'completed'
        checks['all_five_surfaces_visited'] = runtime.visited == {tile['id'] for tile in tools.config['surfaces']}
        checks['panel_shows_all_metrics'] = all(text in tools.metrics_label.text for text in ('Slip L / R', 'Sinkage L / R', 'RTF (~1 s)'))
        checks['all_samples_share_schema'] = all(set(r) == set(runtime.history[0]) for r in runtime.history)
        checks['rtf_finite_positive'] = all(math.isfinite(r['rtf_average']) and r['rtf_average'] > 0 for r in runtime.history[10:])
        checks['rtf_matches_recorded_durations'] = all(abs(r['rtf_average'] - r['rtf_sim_time_s']/r['active_wall_time_s']) < 1e-9 for r in runtime.history[10:])
        checks['rigid_sinkage_near_zero'] = max(r[s+'_sinkage_m'] for r in runtime.history for s in ('left', 'right')) < .002
        exported = tools.export()
        with (exported/'physics.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        checks['csv_row_count'] = len(rows) == len(runtime.history)
        checks['csv_contains_slip_sinkage_rtf'] = all(key in rows[0] for key in (
            'left_slip_ratio', 'right_slip_ratio', 'left_sinkage_m', 'right_sinkage_m',
            'rtf_window', 'rtf_average', 'active_wall_time_s'))
        checks['csv_invalid_slip_explicit_nan'] = rows[0]['left_slip_ratio'] == 'nan'
        manifest = json.loads((PROJECT/'source_manifest.json').read_text())
        checks['original_assets_dynamics_unchanged'] = all(
            hashlib.sha256((PROJECT/path).read_bytes()).hexdigest() == digest
            for path, digest in manifest['unchanged_copies_sha256'].items())
        runtime.save_csv(output/'physics.csv')
        (output/'panel_text.txt').write_text(tools.metrics_label.text + '\n')
        report.update(
            passed=all(checks.values()), export_path=str(exported), samples=len(rows),
            sim_duration_s=runtime.sim_time, active_wall_s=runtime.latest['active_wall_time_s'],
            rtf_average=runtime.latest['rtf_average'], rtf_window=runtime.latest['rtf_window'],
            max_geometric_sinkage_m=max(r[s+'_sinkage_m'] for r in runtime.history for s in ('left','right')),
            final_position_m=[runtime.latest[k] for k in ('x_m','y_m','z_m')])
        print('TELEMETRY_VALIDATION: '+json.dumps(report), flush=True)
    except Exception:
        report['error'] = traceback.format_exc()
        if tools:
            report['runtime_at_failure'] = {'sim_s': tools.runtime.sim_time,
                'playing': tools.runtime.timeline.is_playing(),
                'timeline_s': tools.runtime.timeline.get_current_time(),
                'initialization_attempts': tools.runtime.initialization_attempts,
                'samples': len(tools.runtime.history), 'runtime_error': tools.runtime.error}
        print(report['error'], flush=True)
    finally:
        (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
        if tools:
            tools.runtime.zero()
            tools.runtime.timeline.stop()
            tools.runtime.shutdown()
        await omni.usd.get_context().close_stage_async()
        app.post_quit(0 if report['passed'] else 1)


VALIDATION_TASK = asyncio.ensure_future(validate())
