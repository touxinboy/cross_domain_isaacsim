"""Count failures outside MSGNav/OmniNav; never change a navigation decision.

Uncertain observations and safety interventions are separate from confirmed
errors. Privileged target bounds are accepted only for a completed episode.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile


def verify_baseline_freeze(project):
    root = Path(project).resolve()
    manifest = json.loads((root/'configs/baseline_model_freeze.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = (root/entry['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError('Frozen model source is missing or outside project: '+entry['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('MSGNav/OmniNav baseline freeze differs: '+entry['path'])
    return {'status': 'passed', 'checked_source_files': len(manifest['files']),
            'freeze_manifest_sha256': hashlib.sha256((root/'configs/baseline_model_freeze.json').read_bytes()).hexdigest(),
            'models_modified': False, 'weight_check': 'checkpoint lock metadata unchanged; weights are not loaded by this audit'}


class Evidence:
    def __init__(self, source):
        self.path = Path(source)
        self.zip = ZipFile(self.path) if self.path.is_file() else None
        if self.zip:
            manifest = json.loads(self.zip.read('EXPORT_MANIFEST.json'))
            self.prefix = 'results/semantic/'+manifest['run_id']+'/'
            self.names = {n[len(self.prefix):] for n in self.zip.namelist() if n.startswith(self.prefix)}
        else:
            self.prefix = ''
            self.names = {p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()}

    def read(self, name):
        return self.zip.read(self.prefix+name).decode('utf-8-sig') if self.zip else (self.path/name).read_text(encoding='utf-8-sig')

    def json(self, name, default=None):
        return json.loads(self.read(name)) if name in self.names else default

    def rows(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name)))) if name in self.names else []

    def close(self):
        if self.zip:
            self.zip.close()


def point_distance(xy, path):
    best = math.inf
    for a, b in zip(path, path[1:]):
        vx, vy = b[0]-a[0], b[1]-a[1]
        norm = vx*vx+vy*vy
        t = max(0., min(1., ((xy[0]-a[0])*vx+(xy[1]-a[1])*vy)/norm)) if norm else 0.
        best = min(best, math.hypot(xy[0]-a[0]-t*vx, xy[1]-a[1]-t*vy))
    return best


def bound_distance(xy, bounds):
    lo, hi = bounds
    return math.hypot(max(lo[0]-xy[0], 0., xy[0]-hi[0]), max(lo[1]-xy[1], 0., xy[1]-hi[1]))


def audit_run(source, evaluation=None):
    evidence = Evidence(source)
    try:
        summary = evidence.json('summary.json')
        if not summary or summary.get('phase') != 'finished':
            raise RuntimeError('Audit only a completed episode; no target truth is read during navigation')
        evaluation = evaluation or evidence.json('independent_evaluation.json', {})
        events = {}

        def record(kind, key, layer, assessment, **detail):
            event_id = kind+':'+str(key)
            if event_id not in events:
                events[event_id] = {'event_id': event_id, 'kind': kind, 'layer': layer,
                                    'assessment': assessment, **detail}
            return events[event_id]

        semantic_calls = 0
        for name in sorted(evidence.names):
            if not name.startswith('semantic_') or not name[9:-5].isdigit() or not name.endswith('.json'):
                continue
            reply = evidence.json(name)
            step = int(name[9:-5])
            if not reply.get('ok'):
                record('semantic_worker_failure', step, 'interface', 'observed_failure', step=step,
                       evidence_file=name, error=reply.get('error'))
                continue
            update = reply['result'].get('semantic_update')
            if not update:
                continue
            semantic_calls += 1
            if update.get('msgnav_error'):
                record('msgnav_output_unusable', step, 'semantic_interface', 'observed_failure', step=step,
                       evidence_file=name, error=update['msgnav_error'],
                       attribution='MSGNav response or adapter contract; not automatically a semantic classification error')
            surface = update.get('surface') or {}
            if surface.get('accepted'):
                bounds = evaluation.get('target_bounds_m')
                point = surface.get('world_point_m')
                if bounds is not None and point is not None:
                    distance = bound_distance(point[:2], bounds)
                    if distance > .75:
                        record('target_association_error', step, 'perception_association', 'confirmed_error', step=step,
                               evidence_file=name, selected_rgb=update.get('selected_rgb'),
                               observed_point_m=point, distance_to_true_target_bounds_m=distance,
                               attribution='Qwen grounding/identity and RGB-D association; do not assign to MSGNav without object-level review')
                else:
                    record('target_association_needs_review', step, 'perception_association', 'requires_review',
                           step=step, evidence_file=name)
            else:
                record('target_unconfirmed', step, 'perception', 'neutral_observation', step=step,
                       evidence_file=name, note='Occlusion or no visibility is not automatically a model mistake')

        semantic_events = evidence.json('semantic_events.json', [])
        rejected = set()
        truncated = {}
        for entry in semantic_events:
            kind = entry.get('event')
            step = entry.get('step')
            if kind == 'unsafe_head_rejected':
                rejected.add(step)
                record('waypoint_safety_rejection', step, 'safety_layer', 'intervention', step=step,
                       source_model='OmniNav', evidence_file='semantic_events.json',
                       note='Observed-space rejection; ground-truth review is needed before counting an unsafe model prediction')
            elif kind == 'head_polyline_truncated':
                truncated[step] = float(entry['kept_m'])
                record('waypoint_safety_truncation', step, 'safety_layer', 'intervention', step=step,
                       source_model='OmniNav', original_m=entry['original_m'], kept_m=entry['kept_m'])
            elif kind == 'unvalidated_route_start':
                record('planner_execution_start_recheck', step, 'planning_execution_interface', 'intervention', step=step,
                       evidence_file='semantic_events.json', waypoint_model_called=False,
                       note='Perception retry before inference; the model is not given an unvalidated execution start')

        heads = {}
        for name in sorted(evidence.names):
            if not name.startswith('head_') or not name[5:-5].isdigit() or not name.endswith('.json'):
                continue
            step = int(name[5:-5]);reply = evidence.json(name);heads[step] = reply
            if not reply.get('ok'):
                record('waypoint_worker_failure', step, 'interface', 'observed_failure', step=step,
                       evidence_file=name, error=reply.get('error'), attribution='Model runtime or interface; requires review')
        plans = []
        for entry in evidence.json('events.json', []):
            if entry.get('event') != 'head_accepted':
                continue
            plan = entry['plan'];time_s = float(plan['capture_time_s'])
            step = next((s for s, h in heads.items() if h.get('ok') and
                         abs(float(h['result']['capture_time_s'])-time_s) < 1e-6), None)
            if step is None or step in rejected:
                continue
            path = plan.get('path_xy_m') or [plan['anchor_pose_xyyaw'][:2], plan['goal_xy_m']]
            if step in truncated:
                kept = [path[0]];length = 0.
                for a, b in zip(path, path[1:]):
                    length += math.dist(a, b)
                    if length > truncated[step]+1e-6:
                        break
                    kept.append(b)
                if len(kept) >= 2:
                    path = kept
            plans.append((time_s, step, path))
        plans.sort()
        times = [p[0] for p in plans]
        telemetry = evidence.rows('telemetry.csv')
        physics_times = [float(row['time_s']) for row in telemetry]
        active = None
        incident_index = 0
        controls = evidence.rows('control.csv')
        for row in controls:
            time_s = float(row['time_s'])
            applied = str(row.get('applied', '')).lower() == 'true'
            index = bisect_right(times, time_s+1e-6)-1
            physics_index = max(0, bisect_right(physics_times, time_s+1e-6)-1)
            phase = telemetry[physics_index].get('phase') if telemetry else None
            enabled = applied and index >= 0 and (not phase or phase == 'executing_model_waypoint')
            if not enabled:
                active = None
                continue
            _, step, path = plans[index]
            distance = point_distance([float(row['x_m']), float(row['y_m'])], path)
            if active and active['step'] != step:
                active = None
            if active is None and distance > .20:
                incident_index += 1
                active = record('tracking_deviation', incident_index, 'execution', 'observed_failure',
                                step=step, start_sim_s=time_s, end_sim_s=time_s,
                                max_deviation_m=distance, enter_threshold_m=.20, exit_threshold_m=.10,
                                attribution='Physical controller/plant tracking; not an OmniNav mistake by itself')
            if active:
                active['end_sim_s'] = time_s
                active['max_deviation_m'] = max(active['max_deviation_m'], distance)
                if distance <= .10:
                    active = None

        collision_events = evidence.json('contact_events.json', [])
        if summary.get('reason') == 'obstacle_contact' or collision_events:
            record('obstacle_contact', 1, 'execution', 'observed_failure',
                   sim_time_s=summary.get('sim_time_s'), evidence_file='contact_events.json' if collision_events else 'summary.json',
                   contacts=collision_events, attribution='Collision outcome; algorithm attribution requires trajectory/plant review')
        failures = [e for e in events.values() if e['assessment'] in ['confirmed_error', 'observed_failure']]
        water_evaluated = 'entered_water' in evaluation
        mode_opportunities = int(bool(evaluation['entered_water'])) if water_evaluated else None
        report = {'schema': 'L17.navigation_mistake_counts.v1', 'run_id': summary['run_id'],
                  'task_result': summary.get('reason'), 'post_episode_only': True,
                  'counting_unit': 'one fresh inference failure or one contiguous tracking/collision incident; not per physics frame',
                  'observed_and_confirmed_failure_events': len(failures),
                  'failure_total_note': 'Layer events can share a cause; this sum is not a count of independent model errors',
                  'failure_counts_by_type': dict(Counter(e['kind'] for e in failures)),
                  'failure_counts_by_layer': dict(Counter(e['layer'] for e in failures)),
                  'intervention_counts': dict(Counter(e['kind'] for e in events.values() if e['assessment'] == 'intervention')),
                  'neutral_observation_counts': dict(Counter(e['kind'] for e in events.values() if e['assessment'] == 'neutral_observation')),
                  'pending_review_counts': dict(Counter(e['kind'] for e in events.values() if e['assessment'] == 'requires_review')),
                  'denominators': {'fresh_semantic_updates': semantic_calls, 'waypoint_calls': len(heads),
                                   'accepted_execution_segments': len(plans), 'control_samples': len(controls)},
                  'algorithm_attribution': {'MSGNav_confirmed_semantic_error_count': None,
                                           'OmniNav_confirmed_unsafe_proposal_count': None,
                                           'note': 'Null means not independently annotated; a safety rejection is not a confirmed model error'},
                  'mode_switch_assessment': {'water_entry_opportunities': mode_opportunities,
                                             'switch_error_count': None,
                                             'status': ('pending_post_episode_medium_evaluation' if not water_evaluated else
                                                        'requires_time_aligned_sensor_truth_review' if mode_opportunities else
                                                        'not_evaluated_no_water_entry')},
                  'target_bound_tolerance_m': .75, 'model_outputs_modified': False}
        return report, list(events.values())
    finally:
        evidence.close()


def write_audit(source, output=None, evaluation=None):
    report, events = audit_run(source, evaluation)
    output = Path(output or source)
    output.mkdir(parents=True, exist_ok=True)
    (output/'mistake_counts.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'mistake_events.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in events), encoding='utf-8')
    with (output/'mistake_events.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['event_id', 'kind', 'layer', 'assessment', 'step', 'evidence_json']
        writer = csv.DictWriter(handle, fieldnames=fields);writer.writeheader()
        for event in events:
            writer.writerow({**{k: event.get(k) for k in fields[:-1]}, 'evidence_json': json.dumps(event, ensure_ascii=False)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Completed run directory or evidence ZIP')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_audit(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
