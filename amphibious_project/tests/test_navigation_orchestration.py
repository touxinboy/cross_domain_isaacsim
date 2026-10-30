"""Exercise the actual async orchestration with a CPU event/plant harness.

The harness is NOT PhysX or a trained model. It tests lifecycle, queue, control,
export and cleanup paths that small contract-only tests cannot exercise.
"""
import asyncio
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import uuid

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"scripts"))
from navigation_io import read_json,write_json,waypoint_code_identity
from navigation_loop import NavigationSession
from omninav_worker import process_request
from test_navigation_preparation import CaptureFixture,FixturePredictor


class Timeline:
    def __init__(self):self.playing=False;self.pause_count=0;self.deferred=False;self.pending=None
    def is_playing(self):return self.playing
    def play(self):
        if self.deferred:self.pending=True
        else:self.playing=True
    def pause(self):
        if self.deferred:self.pending=False
        else:self.playing=False
        self.pause_count+=1


class Harness:
    def __init__(self,root):
        self.root=root;self.timeline=Timeline();self.runtime=None;self.fail_export=False
        self.stage_path=root/"scenes/test.usda"
        layer=types.SimpleNamespace(realPath=str(self.stage_path))
        self.stage=types.SimpleNamespace(GetRootLayer=lambda:layer)
        self.predictor=FixturePredictor();self.no_worker=False
        self.restored=False
        self.capture_playing=[];self.inference_playing=[]

    async def update(self):
        await asyncio.sleep(0)
        if self.timeline.pending is not None:
            self.timeline.playing=self.timeline.pending;self.timeline.pending=None
        if self.timeline.playing and self.runtime:self.runtime.on_step(.05)
        if not self.no_worker:
            for path in (self.root/"queue").glob("*.request.json"):
                if not path.with_name(path.name.replace(".request.json",".result.json")).exists():
                    self.inference_playing.append(self.timeline.playing)
                    process_request(self.root,path,self.predictor)

    def modules(self):
        host=self
        class BaseRuntime:
            def __init__(self,project,kind):
                self.project=Path(project);self.timeline=host.timeline;self.stage=host.stage
                self.sim_time=0.;self.clock_epoch=uuid.uuid4().hex;self.latest=None;self.history=[]
                self.error=None;self.body_view=None;self.cfg={"physics_hz":120}
                self.x=-12.;self.y=-23.;self.phase="manual"
                self.commands={"mode":"LAND","tracks":[0.,0.],"thrusters":[0.,0.]}
                host.runtime=self
            def zero(self):self.commands["tracks"]=[0.,0.];self.commands["thrusters"]=[0.,0.]
            def set_commands(self,**command):self.commands=command
            def on_contacts(self,*args):pass
            def on_step(self,dt):
                if not self.timeline.playing:return
                v=sum(self.commands["tracks"])/2
                self.x+=v*dt;self.sim_time+=dt;self.body_view=True
                self.latest={"time_s":self.sim_time,"x_m":self.x,"y_m":self.y,"yaw_rad":0.,
                             "vx_m_s":v,"vy_m_s":0.,"wz_rad_s":0.,"buoyancy_n":0.,
                             "left_normal_n":235.44,"right_normal_n":235.44,"roll_deg":0.,"pitch_deg":0.}
                self.history.append(self.latest.copy())
            def fail(self,phase,error):self.error=phase+error;self.timeline.pause()
            def shutdown(self):self.zero()
            def save_csv(self,path):
                if host.fail_export:raise OSError("simulated telemetry export failure")
                with Path(path).open("w",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(self.history[0]));writer.writeheader();writer.writerows(self.history)
        runtime_module=types.ModuleType("robot_runtime")
        runtime_module.RobotRuntime=BaseRuntime;runtime_module._RUNTIME=None
        runtime_module.get_robot_runtime=lambda:runtime_module._RUNTIME
        def detach():
            if runtime_module._RUNTIME:runtime_module._RUNTIME.shutdown()
            runtime_module._RUNTIME=None;host.runtime=None
        def attach(project,kind):
            runtime_module._RUNTIME=BaseRuntime(project,kind);host.restored=True
            return runtime_module._RUNTIME
        runtime_module.detach_robot_runtime=detach;runtime_module.attach_robot_runtime=attach
        omni=types.ModuleType("omni");kit=types.ModuleType("omni.kit")
        app=types.ModuleType("omni.kit.app");app.get_app=lambda:types.SimpleNamespace(next_update_async=host.update)
        timeline=types.ModuleType("omni.timeline");timeline.get_timeline_interface=lambda:host.timeline
        usd=types.ModuleType("omni.usd");usd.get_context=lambda:types.SimpleNamespace(get_stage=lambda:host.stage)
        omni.kit=kit;kit.app=app;omni.timeline=timeline;omni.usd=usd
        return {"omni":omni,"omni.kit":kit,"omni.kit.app":app,"omni.timeline":timeline,
                "omni.usd":usd,"robot_runtime":runtime_module}

    def rig(self,project):
        fixture=CaptureFixture(project)
        async def capture_all(runtime):
            self.capture_playing.append(self.timeline.playing)
            summary=fixture.capture(runtime.sim_time,runtime.clock_epoch)
            for entry in summary["frame_files"]:
                path=Path(entry["frame_file"]);meta=read_json(path)
                meta["robot_pose_xyyaw"]=[runtime.x,runtime.y,0.];meta["synthetic_test_fixture_only"]=True
                write_json(path,meta)
            return summary
        return types.SimpleNamespace(capture_all=capture_all)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="l17_orchestration_");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        names=["configs/navigation_closed_loop.json","configs/robot_v1.json","configs/river_forest.json",
               "configs/crossing_experiment.json","configs/omninav_checkpoint.lock.json","configs/task.json","configs/navigation_camera.json",
               "scripts/omninav_pointgoal.py","scripts/navigation_contracts.py",
               "third_party/OmniNav/infer_ovon_slowfast/qwen_utils.py"]
        for name in names:
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/name,target)
        cfg=read_json(self.root/"configs/navigation_closed_loop.json")
        cfg.update(worker_queue="queue",settle_s=.1,history_spacing_s=.05,execution_slice_s=.25,max_inference_cycles=12)
        write_json(self.root/"configs/navigation_closed_loop.json",cfg)
        task=read_json(self.root/"configs/task.json");task["scene_file"]="scenes/test.usda";write_json(self.root/"configs/task.json",task)
        expected=next(e["sha256"] for e in read_json(self.root/"configs/omninav_checkpoint.lock.json")["files"] if e["path"]=="config.json")
        ready={"worker_id":"cpu_harness","project":str(self.root),"model_loaded":True,
               "checkpoint_config_sha256":expected,"code_sha256":waypoint_code_identity(self.root)}
        write_json(self.root/"queue/ready.json",ready)
        write_json(self.root/"queue/heartbeat.json",{**ready,"wall_unix_s":time.time()})
        self.host=Harness(self.root)
        self.request={"schema":"L17.navigation_run.v1","run_id":"test_run","mode":"shadow",
                      "start_deadline_wall_unix_s":time.time()+300,
                      "goal":{"kind":"robot_arrival_pose","source":"manual_pointgoal_interface_test","goal_xy_m":[-11.,-23.]},
                      "output_relative":"results/test_run","stop_relative":"stop.json"}

    def run_experiment(self):
        with patch.dict(sys.modules,self.host.modules()):
            spec=importlib.util.spec_from_file_location("l17_orchestration_under_test",PROJECT/"scripts/navigation_experiment.py")
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            real_session=NavigationSession
            module.NavigationSession=lambda *a,**kw:real_session(*a,**kw,allow_test_results=True)
            experiment=module.NavigationExperiment(self.root,self.request,rig_factory=self.host.rig)
            asyncio.run(experiment.run())
        return experiment

    def test_shadow_uses_full_capture_queue_and_export_lifecycle(self):
        experiment=self.run_experiment()
        self.assertIsNone(experiment.error)
        summary=read_json(self.root/"results/test_run/summary.json")
        self.assertEqual(summary["reason"],"shadow_interface_recorded")
        self.assertTrue((self.root/"results/test_run/head_001.json").exists())
        self.assertFalse(read_json(self.root/"results/test_run/shadow_command.json")["applied_to_robot"])
        self.assertTrue(self.host.restored);self.assertFalse(self.host.timeline.playing)

    def test_execute_integrates_repeated_short_steps_and_restores_manual_runtime(self):
        self.request["mode"]="execute"
        experiment=self.run_experiment()
        self.assertIsNone(experiment.error)
        self.assertGreater(experiment.cycles,1)
        self.assertGreater(experiment.runtime.x,-12.)
        self.assertTrue(any(row["applied"] and max(row["left_track"],row["right_track"])>0 for row in experiment.runtime.control_rows))
        self.assertTrue(self.host.restored)
        self.assertEqual(experiment.runtime.commands["tracks"],[0.,0.])

    def test_drive_keeps_heading_alignment_active_until_rotation_finishes(self):
        with patch.dict(sys.modules, self.host.modules()):
            spec=importlib.util.spec_from_file_location('l17_heading_advance_under_test',PROJECT/'scripts/navigation_experiment.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            experiment=module.NavigationExperiment.__new__(module.NavigationExperiment)
            runtime=types.SimpleNamespace(sim_time=0.,enabled=False,stage=self.host.stage,
                                          control_rows=[],zero=lambda:None)
            experiment.runtime=runtime
            async def update():
                await asyncio.sleep(0)
                runtime.sim_time+=.05
                status='heading_alignment' if runtime.sim_time<.3 else 'short_waypoint_reached'
                runtime.control_rows.append({'time_s':runtime.sim_time,'status':status})
            experiment.update=update
            asyncio.run(experiment.advance(.8,drive=True))
            self.assertGreaterEqual(runtime.sim_time,.3)
            self.assertLess(runtime.sim_time,.5)
            self.assertFalse(self.host.timeline.playing)

    def test_deferred_kit_pause_is_applied_before_capture_and_model_inference(self):
        self.host.timeline.deferred=True
        experiment=self.run_experiment()
        self.assertIsNone(experiment.error)
        self.assertTrue(self.host.capture_playing);self.assertTrue(self.host.inference_playing)
        self.assertFalse(any(self.host.capture_playing))
        self.assertFalse(any(self.host.inference_playing))
        waits=[e for e in experiment.session.events if e['event']=='inference_physics_pause_verified']
        self.assertTrue(waits)
        self.assertTrue(all(e['advanced_sim_s']==0 for e in waits))

    def test_export_error_still_zeros_commands_and_releases_runtime(self):
        self.host.fail_export=True
        experiment=self.run_experiment()
        self.assertTrue(experiment.cleanup_errors)
        self.assertTrue(self.host.restored)
        self.assertEqual(experiment.runtime.commands["tracks"],[0.,0.])
        self.assertTrue(read_json(self.root/"results/test_run/summary.json")["cleanup_errors"])

    def test_existing_run_results_are_not_overwritten(self):
        write_json(self.root/"results/test_run/summary.json",{"original":True})
        self.host.timeline.playing=True
        experiment=self.run_experiment()
        self.assertFalse(experiment.output_created)
        self.assertEqual(read_json(self.root/"results/test_run/summary.json"),{"original":True})
        self.assertTrue(self.host.timeline.playing)

    def test_other_stage_is_not_paused_or_controlled(self):
        self.host.stage.GetRootLayer=lambda:types.SimpleNamespace(realPath=str(self.root.parent/"someone_else.usda"))
        self.host.timeline.playing=True
        experiment=self.run_experiment()
        self.assertIn("own L17 robot scene",experiment.error)
        self.assertTrue(self.host.timeline.playing);self.assertIsNone(experiment.runtime)

    def test_stopped_worker_is_rejected_even_with_a_recent_heartbeat(self):
        write_json(self.root/"queue/stopped.json",{"worker_id":"cpu_harness"})
        experiment=self.run_experiment()
        self.assertIn("worker has stopped",experiment.error)
        self.assertIsNone(experiment.runtime)

    def test_inference_failure_exports_error_and_restores_control(self):
        class Broken:
            def predict(self,request):raise RuntimeError("fixture backend failed")
        self.host.predictor=Broken()
        experiment=self.run_experiment()
        self.assertIn("fixture backend failed",experiment.error)
        self.assertTrue(self.host.restored)
        self.assertIsNone(experiment.session.pending)

    def semantic_episode(self, unsafe_start=False):
        import numpy as np
        from contextlib import nullcontext
        from navigation_io import file_sha
        semantic_cfg=read_json(PROJECT/'configs/semantic_navigation.json')
        semantic_cfg.update(max_inference_cycles=12,settle_s=.1,history_spacing_s=.05,execution_slice_s=.5)
        write_json(self.root/'configs/semantic_navigation.json',semantic_cfg)
        state_file=self.root/'observed/state.npy';state_file.parent.mkdir()
        state=np.ones((48,48),np.int8)
        if unsafe_start:state[24,26]=-1  # support cell is free; its full footprint lacks observation
        np.save(state_file,state)
        write_json(state_file.with_name('grid.json'),{'schema':'L17.traversability.v1','source':'perception',
                   'state_file':'state.npy','state_sha256':file_sha(state_file),'origin_xy_m':[-18.,-29.],'resolution_m':.25})
        request={**self.request,'schema':'L17.semantic_navigation_run.v1','mode':'execute',
                 'target_id':semantic_cfg['target_id'],'instruction':semantic_cfg['instruction'],'synthetic_test_fixture_only':True}
        self.host.stage.GetPrimAtPath=lambda path:None;self.host.stage.GetSessionLayer=lambda:None
        pxr=types.ModuleType('pxr');pxr.Usd=types.SimpleNamespace(EditContext=lambda *a:nullcontext());pxr.UsdGeom=types.SimpleNamespace()
        with patch.dict(sys.modules,{**self.host.modules(),'pxr':pxr}):
            spec=importlib.util.spec_from_file_location('navigation_experiment',PROJECT/'scripts/navigation_experiment.py')
            base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
            with patch.dict(sys.modules,{'navigation_experiment':base}):
                spec=importlib.util.spec_from_file_location('l17_semantic_under_test',PROJECT/'scripts/semantic_navigation_experiment.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                real_session=NavigationSession
                module.NavigationSession=lambda *a,**kw:real_session(*a,**kw,allow_test_results=True)
                class HarnessExperiment(module.SemanticNavigationExperiment):
                    scan_attempts=0
                    def semantic_ready(self):return {'worker_id':'cpu_semantic_fixture','synthetic_test_fixture_only':True}
                    async def await_semantic_plan(self,path,observation):
                        return {'semantic_arrival_candidate':self.runtime.x>-11.2,'need_scan':False,'grid_file':'observed/grid.json',
                                'footprint_radius_m':.68,'target_distance_m':abs(-10.-self.runtime.x),'target_surface_xy_m':[-10.,-23.],
                                'route':{'kind':'synthetic_observed_route_fixture','local_goal_xy_m':[-10.,-23.]}}
                    async def scan(self,rig):
                        self.scan_attempts+=1
                        if unsafe_start:self.reason='fixture_perception_retry_recorded'
                        else:await super().scan(rig)
                experiment=HarnessExperiment(self.root,request,rig_factory=self.host.rig)
                asyncio.run(experiment.run())
        return experiment

    def test_semantic_episode_runs_route_head_motion_and_cleanup_in_cpu_harness(self):
        experiment=self.semantic_episode()
        self.assertIsNone(experiment.error)
        self.assertEqual(experiment.reason,'semantic_flat_top_jacket_reached')
        self.assertGreater(experiment.runtime.x,-11.2);self.assertTrue(self.host.restored)
        self.assertTrue((experiment.output/'semantic_events.json').exists())
        self.assertFalse(read_json(experiment.output/'summary.json')['manual_target_coordinate_used'])

    def test_unsupported_start_requests_observation_before_any_waypoint_inference(self):
        experiment=self.semantic_episode(unsafe_start=True)
        self.assertEqual(experiment.scan_attempts,1)
        self.assertEqual(experiment.runtime.x,-12.)
        self.assertEqual(self.host.inference_playing,[])
        self.assertFalse(list(experiment.output.glob('head_*.json')))
        events=read_json(experiment.output/'semantic_events.json')
        self.assertEqual(events[0]['event'],'unvalidated_route_start')
        self.assertFalse(events[0]['waypoint_model_called'])


if __name__=="__main__":unittest.main(verbosity=2)
