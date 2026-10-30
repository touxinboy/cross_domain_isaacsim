"""Verify incident counting and evidence attribution without a model or simulator."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))
from navigation_audit import audit_run,write_audit,verify_baseline_freeze


class NavigationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='l17_audit_');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        def save(name,value):(self.root/name).write_text(json.dumps(value),encoding='utf-8')
        self.save=save
        save('summary.json',{'phase':'finished','run_id':'audit_fixture','reason':'obstacle_contact','sim_time_s':.4})
        save('head_0001.json',{'ok':True,'result':{'capture_time_s':0.}})
        save('events.json',[{'event':'head_accepted','plan':{'capture_time_s':0.,'path_xy_m':[[0.,0.],[2.,0.]]}}])
        save('semantic_events.json',[{'event':'unsafe_head_rejected','step':9}])
        save('semantic_0001.json',{'ok':True,'result':{'semantic_update':{'msgnav_error':'bad\nreply, "x"',
             'surface':{'accepted':True,'world_point_m':[20.,10.,1.]}}}})
        save('semantic_0002.json',{'ok':True,'result':{'semantic_update':{'surface':{'accepted':False}}}})
        save('independent_evaluation.json',{'target_bounds_m':[[1.,1.,0.],[3.,3.,3.]],'entered_water':False})
        lateral=[0.,.25,.28,.30,.05,.32,.35]
        for filename,fields in [('telemetry.csv',['time_s','phase']),('control.csv',['time_s','applied','x_m','y_m'])]:
            with (self.root/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for index,y in enumerate(lateral):
                    row={'time_s':index*.05,'phase':'executing_model_waypoint'} if filename=='telemetry.csv' else {
                        'time_s':index*.05,'applied':True,'x_m':.5,'y_m':y}
                    writer.writerow(row)

    def test_continuous_error_samples_are_counted_as_incidents(self):
        report,events=audit_run(self.root)
        self.assertEqual(report['failure_counts_by_type']['tracking_deviation'],2)
        drift=[e for e in events if e['kind']=='tracking_deviation']
        self.assertAlmostEqual(drift[0]['max_deviation_m'],.30)
        self.assertEqual(report['failure_counts_by_type']['obstacle_contact'],1)

    def test_safety_rejections_and_inconclusive_views_are_not_model_errors(self):
        report,_=audit_run(self.root)
        self.assertEqual(report['intervention_counts']['waypoint_safety_rejection'],1)
        self.assertEqual(report['neutral_observation_counts']['target_unconfirmed'],1)
        self.assertIsNone(report['algorithm_attribution']['OmniNav_confirmed_unsafe_proposal_count'])
        self.assertEqual(report['mode_switch_assessment']['status'],'not_evaluated_no_water_entry')

    def test_target_truth_is_read_only_after_episode_finishes(self):
        self.save('summary.json',{'phase':'running'})
        (self.root/'independent_evaluation.json').write_text('invalid truth content',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'completed episode'):audit_run(self.root)

    def test_missing_medium_evaluation_does_not_claim_no_water_entry(self):
        (self.root/'independent_evaluation.json').unlink()
        report,_=audit_run(self.root)
        self.assertIsNone(report['mode_switch_assessment']['water_entry_opportunities'])
        self.assertEqual(report['mode_switch_assessment']['status'],'pending_post_episode_medium_evaluation')

    def test_model_source_change_is_detected_before_start(self):
        import hashlib
        (self.root/'configs').mkdir()
        source=self.root/'model.py';source.write_text('unchanged model',encoding='utf-8')
        self.save('configs/baseline_model_freeze.json',{'files':[{'path':'model.py','sha256':hashlib.sha256(source.read_bytes()).hexdigest()}]})
        self.assertEqual(verify_baseline_freeze(self.root)['status'],'passed')
        source.write_text('modified model',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'baseline freeze differs'):verify_baseline_freeze(self.root)

    def test_csv_preserves_error_evidence_and_truth_is_not_assigned_to_msgnav(self):
        report=write_audit(self.root)
        self.assertEqual(report['failure_counts_by_type']['target_association_error'],1)
        self.assertIsNone(report['algorithm_attribution']['MSGNav_confirmed_semantic_error_count'])
        with (self.root/'mistake_events.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        event=next(json.loads(row['evidence_json']) for row in rows if row['kind']=='msgnav_output_unusable')
        self.assertEqual(event['error'],'bad\nreply, "x"')


if __name__=='__main__':unittest.main()
