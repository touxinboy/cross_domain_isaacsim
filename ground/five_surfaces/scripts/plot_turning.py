"""Plot measured turning CSVs: python3 scripts/plot_turning.py RESULTS_DIRECTORY."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from turning_experiments import combined_slip_rms


COLORS = dict(Grass='#2e7d32', DrySand='#c48720', WetSand='#1976b8',
              Mud='#8a5140', Riverbed='#8257ad')


def read_rows(path):
    rows = []
    with Path(path).open(newline='') as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def generate_plots(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(directory)
    manifest = json.loads((directory/'experiments.json').read_text())
    trials = []
    for trial in manifest['trials']:
        rows = read_rows(directory/trial['csv'])
        rows = [r for r in rows if r['turn_phase'] == 'turn']
        if rows:
            trials.append((trial, rows))
    if not trials:
        raise ValueError('No measured turning samples available; cannot plot')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.grid':True, 'grid.alpha':.22})
    output = directory/'plots'
    output.mkdir(exist_ok=True)
    artifacts = []

    def save(fig, stem):
        for suffix in ('.png', '.svg'):
            path = output/(stem+suffix)
            fig.savefig(path, dpi=160, bbox_inches='tight')
            artifacts.append(str(path.relative_to(directory)))
        plt.close(fig)

    for mode in ('spin', 'orbit'):
        selected = [(trial, rows) for trial, rows in trials if trial['mode'] == mode]
        if not selected:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), sharex=True, constrained_layout=True)
        labels = ['Longitudinal slip ratio RMS (%)', 'Maximum total sinkage (mm)',
                  'Area-mean normal pressure (kPa)', 'Area-mean shear magnitude (kPa)']
        for trial, rows in selected:
            times = [r['turn_time_s'] for r in rows]
            series = [[100*combined_slip_rms(r) for r in rows],
                      [1000*max(r['left_sinkage_m'], r['right_sinkage_m'])
                       if r['left_sinkage_valid'] or r['right_sinkage_valid'] else np.nan for r in rows],
                      [r['soil_pressure_mean_pa']/1000 for r in rows],
                      [r['soil_shear_stress_mean_pa']/1000 for r in rows]]
            label = trial['surface']+(' ['+trial['result']+']' if trial['result'] != 'completed' else '')
            for ax, values in zip(axes.flat, series):
                ax.plot(times, values, color=COLORS.get(trial['surface']), label=label, lw=1.15)
        for ax, label in zip(axes.flat, labels):
            ax.set_ylabel(label)
        for ax in axes[-1]:
            ax.set_xlabel('Time since turn command (simulation s)')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(('In-place spin' if mode == 'spin' else 'Orbit about a fixed point')+
                     ' | native PhysX + circulating shoes + SCM\nUncalibrated soil; measured samples, no smoothing')
        save(fig, mode+'_comparison')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    for ax, mode in zip(axes, ('spin', 'orbit')):
        for trial, rows in trials:
            if trial['mode'] != mode:
                continue
            cx, cy = trial['center_m']
            color = COLORS.get(trial['surface'])
            ax.plot([r['x_m']-cx for r in rows], [r['y_m']-cy for r in rows],
                    label=trial['surface'], color=color, lw=1.2)
            if mode == 'orbit':
                theta = np.linspace(0, 2*np.pi, 240)
                radius = trial['target_radius_m']
                ax.plot(radius*np.cos(theta), radius*np.sin(theta), 'k:', lw=.7, alpha=.3)
        ax.plot(0, 0, 'k+', ms=9)
        ax.set(xlabel='x relative to fixed center (m)', ylabel='y relative to fixed center (m)',
               title='Spin center drift' if mode == 'spin' else 'Actual orbit and target circle', aspect='equal')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    save(fig, 'trajectories')

    for trial, rows in trials:
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
        t = [r['turn_time_s'] for r in rows]
        keys = ('slip_ratio', 'sinkage_m', 'pressure_mean_pa', 'shear_stress_mean_pa')
        scales = (100., 1000., .001, .001)
        labels = ('Signed longitudinal slip (%)', 'Total sinkage (mm)',
                  'Normal pressure (kPa)', 'Shear magnitude (kPa)')
        for ax, key, scale, label in zip(axes, keys, scales, labels):
            for side, color in (('left', '#1976b8'), ('right', '#db7322')):
                values = [r[side+'_'+key]*scale for r in rows]
                if key in ('slip_ratio', 'sinkage_m'):
                    valid = 'slip_valid' if key == 'slip_ratio' else 'sinkage_valid'
                    values = [v if r[side+'_'+valid] else np.nan for r, v in zip(rows, values)]
                ax.plot(t, values, label=side+' mean' if 'mean' in key else side, color=color, lw=1.)
                if 'mean' in key:
                    ax.plot(t, [r[side+'_'+key.replace('mean', 'max')]*scale for r in rows],
                            '--', color=color, alpha=.6, lw=.8, label=side+' cell max')
            ax.set_ylabel(label)
            ax.legend(fontsize=8, ncol=4)
        axes[-1].set_xlabel('Time since turn command (simulation s)')
        fig.suptitle(f"{trial['surface']} | {trial['mode']} | {trial['result']} | "
                     f"{trial['progress_rad']/(2*np.pi):.2f} measured revolutions")
        save(fig, trial['surface']+'_'+trial['mode'])
    (output/'index.json').write_text(json.dumps({'artifacts':artifacts}, indent=2)+'\n')
    return artifacts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    for name in generate_plots(args.directory):
        print(args.directory/name)
