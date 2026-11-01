"""Plot a continuous five-surface traversal from physics.csv and summary.json."""
import argparse
import json
from pathlib import Path
import threading

import numpy as np

from plot_turning import COLORS, read_rows
from turning_experiments import combined_slip_rms

_PLOT_LOCK = threading.Lock()


def surface_segments(rows):
    """Keep each contiguous COM material interval separate, including revisits."""
    if not rows:
        return []
    starts = [0]+[i for i in range(1, len(rows)) if rows[i]['surface'] != rows[i-1]['surface']]
    return [(rows[start]['surface'], start, stop)
            for start, stop in zip(starts, starts[1:]+[len(rows)])]


def values(rows, key, scale=1., valid=None):
    return np.array([float(r.get(key, np.nan))*scale
                     if valid is None or r.get(valid, 0) else np.nan for r in rows])


def generate_plots(directory):
    # Matplotlib pyplot has process-wide state; serialize repeated GUI exports.
    with _PLOT_LOCK:
        return _generate_plots(directory)


def _generate_plots(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    directory = Path(directory)
    rows = read_rows(directory/'physics.csv')
    summary = json.loads((directory/'summary.json').read_text())
    if not rows:
        raise ValueError('No recorded physics samples')
    t = values(rows, 'time_s')
    segments = surface_segments(rows)
    output = directory/'plots'
    output.mkdir(exist_ok=True)
    artifacts = []
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.grid':True, 'grid.alpha':.22})

    def save(fig, stem):
        for suffix in ('.png', '.svg'):
            path = output/(stem+suffix)
            fig.savefig(path, dpi=160, bbox_inches='tight')
            artifacts.append(str(path.relative_to(directory)))
        plt.close(fig)

    def shade(ax):
        for surface, start, stop in segments:
            end = t[stop] if stop < len(t) else t[-1]
            ax.axvspan(t[start], end, color=COLORS.get(surface, '#777777'), alpha=.09, zorder=0)
            if start:
                ax.axvline(t[start], color='#666666', ls=':', lw=.7, alpha=.6)

    surfaces = list(dict.fromkeys(surface for surface, _, _ in segments))
    handles = [Patch(facecolor=COLORS.get(s, '#777777'), alpha=.25, label=s) for s in surfaces]
    result = summary.get('result', 'unknown')
    model = summary.get('soil_model', 'unknown')
    model_label = 'SCM soil (uncalibrated)' if model.startswith('scm') else 'Rigid ground baseline'
    title = f'Continuous five-surface traversal | {result} | {model_label}'

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True, constrained_layout=True)
    metrics = [('slip_ratio', 100., 'Signed longitudinal slip (%)', 'slip_valid'),
               ('sinkage_m', 1000., 'Total sinkage (mm)', 'sinkage_valid'),
               ('pressure_mean_pa', .001, 'Area-mean normal pressure (kPa)', None),
               ('shear_stress_mean_pa', .001, 'Area-mean shear magnitude (kPa)', None)]
    for ax, (key, scale, label, valid) in zip(axes.flat, metrics):
        shade(ax)
        available = False
        for side, color in (('left', '#1565c0'), ('right', '#df6e21')):
            y = values(rows, side+'_'+key, scale, side+'_'+valid if valid else None)
            ax.plot(t, y, color=color, ls='-' if side == 'left' else '--', lw=1., label=side)
            available |= bool(np.isfinite(y).any())
        if not available:
            ax.text(.5, .5, 'Not recorded / unavailable', ha='center', transform=ax.transAxes)
        ax.set_ylabel(label)
    axes[0, 0].legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('Simulation time (s)')
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.04), ncol=len(handles), fontsize=9)
    fig.suptitle(title+'\nShading: material under chassis COM; contact patches can span boundaries. Raw samples.')
    save(fig, 'straight_timeseries')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    labels = ['Longitudinal slip ratio RMS (%)', 'Maximum total sinkage (mm)',
              'Area-mean normal pressure (kPa)', 'Area-mean shear magnitude (kPa)']
    visits = {}
    for surface, start, stop in segments:
        part = rows[start:stop]
        visits[surface] = visits.get(surface, 0)+1
        label = surface+(f' visit {visits[surface]}' if visits[surface] > 1 else '')
        elapsed = values(part, 'time_s')-t[start]
        data = [[100*combined_slip_rms(r) for r in part],
                [max(r['left_sinkage_m'], r['right_sinkage_m'])*1000
                 if r['left_sinkage_valid'] or r['right_sinkage_valid'] else np.nan for r in part],
                values(part, 'soil_pressure_mean_pa', .001),
                values(part, 'soil_shear_stress_mean_pa', .001)]
        for ax, y in zip(axes.flat, data):
            ax.plot(elapsed, y, color=COLORS.get(surface), label=label, lw=1.1)
    for ax, label in zip(axes.flat, labels):
        ax.set_ylabel(label)
        ax.set_xlabel('Time since COM entered this material (s)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Material segments of the same continuous traversal\nDifferent entry states and segment lengths; not independent fresh-soil trials.')
    save(fig, 'straight_surface_comparison')

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for ax, prefix, label in zip(axes, ('soil_pressure_', 'soil_shear_stress_'),
                                 ('Normal pressure (kPa)', 'Shear magnitude (kPa)')):
        shade(ax)
        available = False
        for key, color, style in (('mean_pa', '#1565c0', '-'), ('max_pa', '#df6e21', '--')):
            y = values(rows, prefix+key, .001)
            ax.plot(t, y, style, color=color, lw=1., label='area mean' if key == 'mean_pa' else 'cell maximum')
            available |= bool(np.isfinite(y).any())
        if not available:
            ax.text(.5, .5, 'Not recorded / unavailable', ha='center', transform=ax.transAxes)
        ax.set_ylabel(label)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('Simulation time (s)')
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.04), ncol=len(handles), fontsize=9)
    fig.suptitle('Continuous traversal | final contact stresses: area means and cell maxima')
    save(fig, 'straight_stress_peaks')
    (output/'straight_index.json').write_text(json.dumps({
        'artifacts':artifacts, 'result':result,
        'material_intervals':[dict(surface=s, first_sample=a, stop_sample=b,
                                   start_time_s=float(t[a]), end_time_s=float(t[b] if b<len(t) else t[-1]))
                              for s,a,b in segments],
        'notes':['COM material labels do not imply a single-material contact patch at boundaries.',
                 'No smoothing or zero-filling of unavailable values.',
                 'Material segment overlays use different entry states from one continuous run.']}, indent=2)+'\n')
    return artifacts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    for name in generate_plots(args.directory):
        print(args.directory/name)
