"""Post-episode evaluation only: privileged target geometry never feeds planning."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from navigation_io import own_path, read_json, write_json, file_sha

PROJECT = Path(__file__).resolve().parents[1]


def bound_distance(xy, bounds):
    low, high = bounds
    return math.hypot(max(low[0]-xy[0], 0., xy[0]-high[0]), max(low[1]-xy[1], 0., xy[1]-high[1]))


def evaluate(output):
    # This guard precedes every read of privileged target configuration/bounds.
    if not (output/'summary.json').is_file():
        raise RuntimeError('Evaluation is permitted only after the semantic episode has finished')
    summary = read_json(output/'summary.json')
    targets = read_json(PROJECT/'configs/jacket_targets_v2.json')
    target = next(t for t in targets['targets'] if t['shape'] == 'flat_top')
    verification_path = PROJECT/'results/jacket_targets_v2/verification.json'
    verification = read_json(verification_path)
    bounds = verification['targets'][target['id']]['bounds_m']
    with (output/'telemetry.csv').open(newline='') as handle:
        telemetry = list(csv.DictReader(handle))
    with (output/'control.csv').open(newline='') as handle:
        controls = list(csv.DictReader(handle))
    if not telemetry:
        raise RuntimeError('No measured physical trajectory was saved')
    first, final = telemetry[0], telemetry[-1]
    xy = lambda row: [float(row[k]) for k in ['x_m', 'y_m']]
    path = sum(math.dist(xy(a), xy(b)) for a, b in zip(telemetry, telemetry[1:]))
    final_distance = bound_distance(xy(final), bounds)
    stopped = math.hypot(float(final['vx_m_s']), float(final['vy_m_s'])) < .10
    grounded = float(final['left_normal_n'])+float(final['right_normal_n']) > .2*48*9.81
    switches = sum(a['mode'] != b['mode'] for a, b in zip(controls, controls[1:]))
    heads = [read_json(f) for f in sorted(output.glob('head_*.json'))]
    semantics = [read_json(f) for f in sorted(output.glob('semantic_[0-9]*.json'))]
    events = read_json(output/'events.json')
    pauses = [e for e in events if e.get('event') == 'inference_physics_pause_verified']
    checks = {
        'semantic_arrival_declared': bool(summary.get('semantic_visual_arrival_verified')),
        'stopped': stopped,
        'supported_on_land': grounded,
        'outside_target_bounds_with_robot_clearance': final_distance >= .68,
        'within_3_5m_of_structure_bounds': final_distance <= 3.5,
        'no_contact_or_attitude_abort': summary['reason'] not in ['obstacle_contact', 'attitude_limit', 'speed_limit'],
        'no_cleanup_errors': not summary.get('cleanup_errors'),
    }
    evaluation = {
        'schema': 'L17.semantic_post_episode_evaluation.v1',
        'run_id': summary['run_id'], 'result': summary['reason'],
        'independent_task_success': all(checks.values()), 'checks': checks,
        'privileged_target_data_read_only_after_episode': True,
        'target_id': target['id'], 'target_bounds_m': bounds,
        'target_bounds_evidence_sha256': file_sha(verification_path),
        'initial_xy_m': xy(first), 'final_xy_m': xy(final),
        'physical_path_length_m': path, 'net_displacement_m': math.dist(xy(first), xy(final)),
        'initial_distance_to_structure_bounds_m': bound_distance(xy(first), bounds),
        'final_distance_to_structure_bounds_m': final_distance,
        'final_robot_clearance_from_bounds_m': final_distance-.68,
        'elapsed_sim_s': float(final['time_s'])-float(first['time_s']),
        'physics_samples': len(telemetry), 'control_samples': len(controls),
        'mode_control_samples': dict(Counter(row['mode'] for row in controls)),
        'mode_switches': switches,
        'max_buoyancy_n': max(float(row['buoyancy_n']) for row in telemetry),
        'entered_water': any(float(row['buoyancy_n']) > .15*48*9.81 for row in telemetry),
        'free_float_observed': any(float(row['buoyancy_n']) > .72*48*9.81 and float(row['left_normal_n'])+float(row['right_normal_n']) < .08*48*9.81 for row in telemetry),
        'max_speed_m_s': max(math.hypot(float(row['vx_m_s']), float(row['vy_m_s'])) for row in telemetry),
        'max_abs_roll_deg': max(abs(float(row['roll_deg'])) for row in telemetry),
        'max_abs_pitch_deg': max(abs(float(row['pitch_deg'])) for row in telemetry),
        'semantic_decisions': len(semantics), 'dedicated_head_calls': len(heads),
        'real_dedicated_head_calls': sum(bool(d.get('model_inference_run')) for d in heads),
        'head_forward_latency_s': [d.get('result', {}).get('latency_s') for d in heads],
        'head_worker_round_trip_s': [d.get('worker_wall_s') for d in heads],
        'max_sim_advance_during_head_inference_s': max([e['advanced_sim_s'] for e in pauses] or [0.]),
        'planner_used_manual_target_coordinate': summary['manual_target_coordinate_used'],
        'planner_used_hidden_scene_map': summary['hidden_scene_map_used_for_planning'],
        'pose_or_velocity_overwrite_during_navigation': summary['pose_or_velocity_overwrite_during_navigation'],
        'observed_target_point_distance_to_true_bounds_m': bound_distance(summary['last_observed_target_xy_m'], bounds) if summary.get('last_observed_target_xy_m') else None,
        'error': summary.get('error'),
        'scope': 'MSGNav decision module + Qwen3-VL + observed RGB-D geometry/heuristic water map + OmniNav trained head + contact/buoyancy reference execution; paused inference; simulator self-pose baseline',
    }
    write_json(output/'independent_evaluation.json', evaluation)
    from navigation_audit import write_audit
    write_audit(output, evaluation=evaluation)
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Own results/semantic run directory; defaults to current run')
    args = parser.parse_args()
    relative = args.run or read_json(PROJECT/'.runtime/semantic/current_request.json')['output_relative']
    print(json.dumps(evaluate(own_path(PROJECT, relative)), indent=2))


if __name__ == '__main__':
    main()
