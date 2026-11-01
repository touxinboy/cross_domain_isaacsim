"""Recompute the interpretation tables and copy original measured plots.

No simulation or smoothing. Inputs are the native CSVs cited by the document.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'scripts'))
from plot_turning import read_rows
from turning_experiments import combined_slip_rms


def stats(rows):
    if not rows:
        raise ValueError('Empty analysis window')

    def a(key):
        return np.array([r[key] for r in rows], dtype=float)

    def mean(x):
        x = np.asarray(x, dtype=float)
        return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else None

    slip = np.array([100*combined_slip_rms(r) for r in rows])
    sinkage = np.maximum(a('left_sinkage_m'), a('right_sinkage_m'))*1000
    load = a('left_normal_n')+a('right_normal_n')
    lateral = np.sqrt((a('left_normal_n')*a('left_lateral_slip_rms_m_s')**2+
                       a('right_normal_n')*a('right_lateral_slip_rms_m_s')**2)/np.maximum(load, 1e-12))
    return dict(samples=len(rows), start_s=rows[0]['time_s'], end_s=rows[-1]['time_s'],
                slip_rms_percent_mean=mean(slip), slip_rms_percent_p95=float(np.nanpercentile(slip,95)),
                sinkage_mm_mean=mean(sinkage), sinkage_mm_max=float(sinkage.max()),
                pressure_kpa_mean=mean(a('soil_pressure_mean_pa')/1000),
                pressure_cell_peak_kpa=float(np.nanmax(a('soil_pressure_max_pa'))/1000),
                shear_kpa_mean=mean(a('soil_shear_stress_mean_pa')/1000),
                shear_cell_peak_kpa=float(np.nanmax(a('soil_shear_stress_max_pa'))/1000),
                area_m2_mean=mean(a('scm_contact_area_m2')), normal_n_mean=mean(load),
                lateral_slip_rms_m_s_mean=mean(lateral),
                speed_m_s_mean=mean(np.hypot(a('vx_m_s'),a('vy_m_s'))),
                yaw_rate_rad_s_mean=mean(a('wz_rad_s')),
                net_longitudinal_traction_n_mean=mean(a('left_traction_n')+a('right_traction_n')),
                side_traction_abs_sum_n_mean=mean(np.abs(a('left_traction_n'))+np.abs(a('right_traction_n'))),
                side_means={side:{key:mean(a(side+'_'+key)) for key in (
                    'belt_m_s','patch_vx_m_s','slip_ratio','traction_n','normal_n')}
                    for side in ('left','right')})


def analyze(straight, turning, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((turning/'experiments.json').read_text())
    summary = json.loads((straight/'summary.json').read_text())
    hashes = {}

    def read(path):
        hashes[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_rows(path)

    rows = read(straight/'physics.csv')
    straight_stats = {}
    for tile in summary['ground_config']['surfaces']:
        interior = [r for r in rows if r['time_s'] >= 3. and
                    tile['x_min_m']+.7 <= r['x_m'] <= tile['x_max_m']-.7 and
                    r['left_track_command_m_s'] > .4 and r['right_track_command_m_s'] > .4 and
                    np.hypot(r['vx_m_s'],r['vy_m_s']) > .2]
        straight_stats[tile['id']] = stats(interior)

    trials = []
    for trial in manifest['trials']:
        all_rows = read(turning/trial['csv'])
        motion = [r for r in all_rows if r['turn_phase'] == 'turn']
        tail = [r for r in motion if r['turn_time_s'] >= 3.]
        entry = dict(surface=trial['surface'], mode=trial['mode'], result=trial['result'],
                     whole_motion=stats(motion), after_3s=stats(tail),
                     motion_duration_s=motion[-1]['turn_time_s'],
                     completed_revolutions=trial['progress_rad']/(2*np.pi),
                     final_sinkage_mm=1000*max(motion[-1]['left_sinkage_m'],motion[-1]['right_sinkage_m']),
                     final_rut_mm=1000*max(motion[-1]['left_rut_depth_m'],motion[-1]['right_rut_depth_m']),
                     max_center_drift_mm=1000*max(r['turn_actual_radius_m'] for r in motion)
                         if trial['mode']=='spin' else None,
                     max_radius_error_mm=1000*max(abs(r['turn_radius_error_m']) for r in motion)
                         if trial['mode']=='orbit' else None,
                     final_radius_m=motion[-1]['turn_actual_radius_m'])
        trials.append(entry)

    events = {}
    for name, sample in (
        ('landing_pressure_peak', max((r for r in rows if r['time_s']<1), key=lambda r:r['soil_pressure_mean_pa'] if np.isfinite(r['soil_pressure_mean_pa']) else -1)),
        ('startup_slip_peak', max((r for r in rows if 1<=r['time_s']<3), key=lambda r:combined_slip_rms(r) if np.isfinite(combined_slip_rms(r)) else -1)),
        ('braking_slip_peak', max((r for r in rows if r['left_track_command_m_s']==0 and r['time_s']>30), key=lambda r:combined_slip_rms(r) if np.isfinite(combined_slip_rms(r)) else -1))):
        keys=('time_s','x_m','vx_m_s','vz_m_s','left_track_command_m_s','left_belt_m_s',
              'left_patch_vx_m_s','left_slip_velocity_m_s','left_slip_ratio','right_slip_ratio','soil_pressure_mean_pa',
              'soil_pressure_max_pa','soil_shear_stress_mean_pa','soil_shear_stress_max_pa')
        events[name]={k:sample[k] if np.isfinite(sample[k]) else None for k in keys}
    sources = [(straight, name) for name in ('straight_timeseries','straight_surface_comparison','straight_stress_peaks')]
    sources += [(turning, name) for name in ('spin_comparison','orbit_comparison','trajectories','Grass_spin','Grass_orbit')]
    figures = {}
    for directory, name in sources:
        source = directory/'plots'/(name+'.png')
        shutil.copyfile(source, output/source.name)
        figures[source.name] = dict(source=str(source.relative_to(PROJECT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    data = dict(date='2026-11-01', straight_source=str(straight.relative_to(PROJECT)),
                turning_source=str(turning.relative_to(PROJECT)),
                definitions=dict(straight='t>=3 s; COM >=0.7 m from both X boundaries of its material; both commands >0.4 m/s; horizontal speed >0.2 m/s',
                                 whole_motion='turn_phase == turn, settling excluded',
                                 after_3s='turn_phase == turn and turn_time_s >= 3',
                                 time_average='Arithmetic sample mean; uniform 120 Hz simulation sampling; NaNs excluded',
                                 sinkage='Instantaneous maximum total sinkage of both sides, then averaged over time; not mean depth over the contact area',
                                 stress='Area-weighted spatial mean each step, then time mean; shear uses magnitude before spatial averaging'),
                csv_sha256=hashes, figures=figures, straight_interior=straight_stats,
                straight_events=events, turning=trials)
    (output/'analysis_data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--straight', type=Path, default=PROJECT/'results/five_surfaces/run_20261101_132512_924339')
    parser.add_argument('--turning', type=Path, default=PROJECT/'results/turning/run_20261101_123133_262831')
    parser.add_argument('--output', type=Path, default=PROJECT/'docs/images/experiment_curves_20261101')
    args=parser.parse_args()
    analyze(args.straight.resolve(),args.turning.resolve(),args.output.resolve())
    print(args.output/'analysis_data.json')
