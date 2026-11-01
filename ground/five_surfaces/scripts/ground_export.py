"""Save immutable ground telemetry snapshots before asynchronous plotting."""
import asyncio
import json
import math
from pathlib import Path
import traceback

from ground_telemetry import telemetry_metadata
from turning_experiments import METRIC_DEFINITIONS


def save_ground_run(runtime, output):
    if not runtime.history:
        return None
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    runtime.save_csv(output/'physics.csv')
    row=runtime.latest
    summary={'result':runtime.route_reason or 'manual_export',
             'visited_surfaces':sorted(runtime.visited),
             'physics_samples':len(runtime.history),
             'final_position_m':[row[k] for k in ['x_m','y_m','z_m']],
             'soil_model':'scm_style_bekker_vector_janosi' if runtime.scm_enabled else 'unchanged_L17_rigid_ground',
             'chassis_mass_and_inertia_unchanged':True,
             'nominal_track_dimensions_preserved':True,
             'robot_physics_unchanged':not runtime.scm_enabled,
             'telemetry':telemetry_metadata(),
             'final_metrics':{key:value for key,value in row.items()
                              if key.startswith(('rtf_', 'active_wall_', 'left_slip', 'right_slip',
                                                 'left_sinkage', 'right_sinkage', 'left_rut', 'right_rut',
                                                 'left_belt_', 'right_belt_', 'left_sprocket_', 'right_sprocket_',
                                                 'left_pressure_', 'right_pressure_', 'left_shear_', 'right_shear_',
                                                 'left_grouser_', 'right_grouser_', 'soil_', 'scm_', 'native_track_'))}}
    # JSON uses null for unavailable metrics; CSV retains explicit NaN.
    summary['final_metrics']={key:(None if isinstance(value,float) and not math.isfinite(value) else value)
                              for key,value in summary['final_metrics'].items()}
    if runtime.scm_enabled:
        runtime.terrain.save(output/'soil_state.npz')
        summary['scm_settings'] = runtime.scm_settings
        summary['track_settings'] = runtime.track_settings
        summary['telemetry'].update(METRIC_DEFINITIONS)
        summary['telemetry'].update(sinkage='maximum h0 minus distributed track-envelope ray intersection height in compressed cells',
            sinkage_scope='SCM total geometric sinkage relative to initial soil surface',
            slip_source='normal-force-weighted distributed contact velocities and filtered equivalent belt speed',
            lateral_slip='normal-force-weighted RMS transverse velocity over distributed contact cells; m/s',
            slip_distance='integral of normal-force-weighted abs local longitudinal slip velocity during contact; m',
            rut_depth='plastic sinkage, retained on unload; no bulk soil flow',
            normal_force='sum of SCM normal forces; native_track_normal_n records rigid contact separately')
        if runtime.track_motion:
            summary['telemetry'].update(
                track_geometry='moving plates and grousers follow a fixed loop; no independent link dynamics',
                sinkage='maximum h0 minus moving shoe/grouser ray intersection height in compressed cells',
                slip_source='distributed material contact velocities including shoe translation and rotation',
                belt_phase='integral of filtered belt speed; shared by render geometry and soil contact',
                sprocket_angle='derived from belt travel and pitch radius, not a physical encoder')
    summary['ground_config'] = runtime.ground_config
    summary['telemetry'].update(
        time_axis='Simulation time since runtime reset/play, including settling and braking.',
        completion='Automatic route: COM reaches goal_x_m, then speed falls below 0.05 m/s after at least 0.6 s braking; result records safety stops separately.',
        reset='Export does not reset soil. Automatic traversal resets once at route start; no reset at material boundaries.',
        surface_labels='Material under chassis COM; distributed contact patches may span material boundaries.',
        comparison_scope='Overlaid material segments come from one continuous traversal with different entry states and lengths.')
    summary['route_start_time_s'] = runtime.route_start_s
    summary['plots_state'] = 'pending'
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return output


async def plot_ground_run(output):
    output = Path(output)
    summary = json.loads((output/'summary.json').read_text())
    try:
        from plot_straight import generate_plots
        summary['plots'] = await asyncio.to_thread(generate_plots, output)
        summary['plots_state'] = 'ready'
        summary.pop('plot_error', None)
    except Exception:
        summary['plots_state'] = 'failed'
        summary['plot_error'] = traceback.format_exc()
    temporary = output/'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2)+'\n')
    temporary.replace(output/'summary.json')
    return summary
