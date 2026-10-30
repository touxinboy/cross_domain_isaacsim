"""Offline contracts and control replay; no Isaac, trained weights or GPU inference."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"scripts"))
from approach_planning import ObservedGrid
from navigation_execution import PhysicalNavigationBridge
from navigation_io import file_sha, read_json, validate_goal, write_json
from navigation_loop import NavigationSession
from navigation_observations import ObservationBuffer
from omninav_pointgoal import prepare_request
from omninav_worker import process_request
from resolve_navigation_depth import resolve, selected_frame


class CaptureFixture:
    def __init__(self,root):self.root=root

    def capture(self,time_s,epoch="epoch1"):
        entries=[]
        for index,name in enumerate(["front","left","right"]):
            folder=self.root/f"capture_{time_s}"/name; folder.mkdir(parents=True,exist_ok=True)
            rgb=folder/"rgb.png";depth=folder/"depth.npy"
            Image.new("RGB",(24,16),(int(time_s)*10,index*50,100)).save(rgb)
            np.save(depth,np.full((16,24),3.,dtype=np.float32))
            meta={"schema":"L17.rgbd.v1","frame_id":name+":"+str(time_s),"camera_role":"robot_mounted",
                  "width":24,"height":16,"rgb_file":rgb.name,"depth_file":depth.name,
                  "rgb_sha256":file_sha(rgb),"depth_sha256":file_sha(depth),"intrinsics":[[10,0,12],[0,10,8],[0,0,1]],
                  "world_from_optical":np.eye(4).tolist(),"optical_axes":"x_right_y_down_z_forward",
                  "depth_type":"distance_to_image_plane","depth_unit":"m","render_time_s":time_s+100,
                  "pose_time_s":time_s+100,"controller_time_s":time_s,"controller_epoch":epoch,"robot_pose_xyyaw":[0.,0.,0.]}
            write_json(folder/"frame.json",meta)
            entries.append({"name":name,"frame_file":str(folder/"frame.json")})
        return {"frame_files":entries}


class TestDepthBindingAndHistory(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="l17_depth_prepare_");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.fixture=CaptureFixture(self.root);self.buffer=ObservationBuffer(self.root)

    def complete(self):
        for t in range(1,6):self.buffer.append_capture(self.fixture.capture(float(t)))
        return self.buffer.observation("Find the observed structure")

    def decision(self):
        return {"source":"MSGNav_original_decision_module","choice_type":"image","capture_time_s":5.,
                "evidence":{"egocentric_index":0},"raw_response":"image 0, structure"}

    def test_distinct_physical_history_builds_real_seven_image_contract(self):
        obs=self.complete();request=prepare_request(obs,[2.,0.],self.root)
        self.assertEqual(len(request["images"]),7)
        self.assertEqual([h["capture_time_s"] for h in obs["pointgoal"]["history"]],[1.,2.,3.,4.])
        self.assertEqual(obs["capture_time_s"],5.)
        self.assertEqual(obs["pointgoal"]["tri_view"]["left"],obs["egocentric_images"][0])

    def test_old_image_pixels_are_rejected_on_new_depth(self):
        obs=self.complete()
        with self.assertRaisesRegex(ValueError,"another RGB frame"):
            resolve(obs,self.decision(),{"visible":True,"image_sha256":"OLD_IMAGE","point_uv":[12,8]},self.root)

    def test_selected_robot_image_becomes_surface_not_drive_goal(self):
        obs=self.complete();_,frame=selected_frame(obs,self.decision(),self.root)
        grounding={"visible":True,"format_valid":True,"image_sha256":frame.metadata["rgb_sha256"],"point_uv":[12,8]}
        result=resolve(obs,self.decision(),grounding,self.root)
        np.testing.assert_allclose(result["surface"]["world_point_m"],[0,0,3])
        self.assertFalse(result["goal_resolved"]);self.assertIsNone(result["goal_xy_m"])
        with self.assertRaisesRegex(ValueError,"arrival pose"):validate_goal(result["surface"])

    def test_reference_camera_and_duplicate_time_do_not_form_robot_history(self):
        summary=self.fixture.capture(1.)
        path=Path(summary["frame_files"][0]["frame_file"])
        meta=read_json(path);meta["camera_role"]="fixed_reference_for_sensor_validation";write_json(path,meta)
        with self.assertRaisesRegex(ValueError,"diagnostic"):self.buffer.append_capture(summary)
        meta["camera_role"]="robot_mounted";write_json(path,meta);self.buffer.append_capture(summary)
        with self.assertRaisesRegex(ValueError,"duplicating"):self.buffer.append_capture(summary)

    def test_runtime_reset_cannot_mix_old_history(self):
        self.buffer.append_capture(self.fixture.capture(1.))
        with self.assertRaisesRegex(ValueError,"runtime reset"):self.buffer.append_capture(self.fixture.capture(2.,"epoch2"))

    def test_modified_current_image_is_rejected_before_inference(self):
        obs=self.complete()
        Image.new("RGB",(24,16),(255,0,0)).save(self.root/obs["pointgoal"]["tri_view"]["front"])
        with self.assertRaisesRegex(ValueError,"modified"):prepare_request(obs,[2.,0.],self.root)

    def test_saved_view_without_depth_cannot_silently_use_another_frame(self):
        obs=self.complete();del obs["egocentric_rgbd_frames"]
        with self.assertRaisesRegex(ValueError,"no matching RGB-D"):selected_frame(obs,self.decision(),self.root)


class TestObservedApproach(unittest.TestCase):
    def surface(self):return {"accepted":True,"camera_role":"robot_mounted","world_point_m":[8.,5.,1.]}

    def test_reachable_standoff_is_outside_object(self):
        grid=ObservedGrid(np.ones((44,44),dtype=np.int8),[0,0],.25)
        goal=grid.approach(self.surface(),[2.,5.],"land")
        self.assertTrue(goal["approach_validated"])
        self.assertGreaterEqual(goal["standoff_m"],1.5)
        self.assertLessEqual(goal["standoff_m"],3.)
        self.assertNotEqual(goal["goal_xy_m"],self.surface()["world_point_m"][:2])
        self.assertTrue(all(grid.segment_is_safe(a,b) for a,b in zip(goal["observed_path_xy_m"],goal["observed_path_xy_m"][1:])))

    def test_unknown_barrier_is_not_free_space(self):
        state=np.ones((44,44),dtype=np.int8);state[:,18:23]=-1
        grid=ObservedGrid(state,[0,0],.25)
        with self.assertRaisesRegex(ValueError,"No observed reachable"):grid.approach(self.surface(),[2.,5.],max_standoff=2.)
        self.assertFalse(grid.segment_is_safe([2.,5.],[8.,5.]))

    def test_footprint_rejects_a_too_narrow_corridor(self):
        state=np.zeros((30,30),dtype=np.int8);state[13:17,:]=1
        grid=ObservedGrid(state,[0,0],.25)
        self.assertFalse(grid.safe.any())


class FixturePredictor:
    """Deterministic CPU transport fixture, NOT trained OmniNav inference."""
    def predict(self,request):
        return {"source":"OmniNav_Slowfast_dedicated_head","capture_time_s":request["capture_time_s"],
                "anchor_pose_xyyaw":request["anchor_pose_xyyaw"],"waypoints_normalized":[[0.,2.]]*5,
                "turn_angles_rad":[0.]*5,"test_fixture_only":True}


class TestWorkerControlReplay(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="l17_worker_replay_");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        buffer=ObservationBuffer(self.root);fixture=CaptureFixture(self.root)
        for t in range(1,6):buffer.append_capture(fixture.capture(float(t)))
        self.obs=buffer.observation("Manual pointgoal interface fixture")
        self.obs_path=self.root/"observation.json";write_json(self.obs_path,self.obs)
        class DryField:
            water_z=0.
            def mesh_height(self,x,y):return 1.
        self.bridge=PhysicalNavigationBridge(read_json(PROJECT/"configs/robot_v1.json"),DryField(),read_json(PROJECT/"configs/crossing_experiment.json"))
        self.session=NavigationSession(self.root,"epoch1",self.bridge,worker_timeout_s=10,allow_test_results=True)
        self.goal={"kind":"robot_arrival_pose","source":"manual_pointgoal_interface_test","goal_xy_m":[2.,0.]}
        self.row={"time_s":5.,"x_m":0.,"y_m":0.,"yaw_rad":0.,"vx_m_s":0.,"vy_m_s":0.,
                  "wz_rad_s":0.,"buoyancy_n":0.,"left_normal_n":235.44,"right_normal_n":235.44}

    def submit(self):return self.session.submit(self.obs_path,self.obs,self.goal,"queue",100.)

    def test_file_request_through_worker_and_actuator_allocator(self):
        path=self.submit();reply=process_request(self.root,path,FixturePredictor())
        self.session.accept(reply,self.row,100.5)
        result=self.bridge.tick(self.row,.05)
        self.assertEqual(result["status"],"tracking")
        self.assertGreater(min(result["command"]["tracks"]),0.)
        self.assertEqual(result["command"]["thrusters"],[0.,0.])
        self.assertEqual(self.row["x_m"],0.)

    def test_previous_runtime_reply_is_rejected_and_clears_commands(self):
        reply=process_request(self.root,self.submit(),FixturePredictor());reply["controller_epoch"]="OLD_EPOCH"
        with self.assertRaisesRegex(ValueError,"identity mismatch"):self.session.accept(reply,self.row,100.5)
        self.assertEqual(self.bridge.tick(self.row,.05)["command"]["tracks"],[0.,0.])

    def test_model_error_is_preserved_and_does_not_become_a_default_plan(self):
        class Broken:
            def predict(self,request):raise RuntimeError("recorded model failure")
        reply=process_request(self.root,self.submit(),Broken())
        self.assertFalse(reply["ok"])
        self.assertIn("recorded model failure",reply["error"])
        with self.assertRaisesRegex(RuntimeError,"inference failed"):self.session.accept(reply,self.row,100.5)
        self.assertIsNone(self.bridge.plan)

    def test_slow_reply_and_large_pose_drift_are_rejected(self):
        reply=process_request(self.root,self.submit(),FixturePredictor())
        self.row["time_s"]=7.1
        with self.assertRaisesRegex(ValueError,"stale"):self.session.accept(reply,self.row,100.5)
        self.row["time_s"]=5.;self.row["x_m"]=.5
        reply=process_request(self.root,self.submit(),FixturePredictor())
        with self.assertRaisesRegex(ValueError,"capture anchor"):self.session.accept(reply,self.row,100.5)

    def test_worker_timeout_clears_pending_and_plan(self):
        self.submit();self.assertTrue(self.session.check_timeout(111.))
        self.assertIsNone(self.session.pending);self.assertIsNone(self.bridge.plan)

    def test_fixture_reply_is_never_accepted_by_the_live_session_default(self):
        self.session.allow_test_results=False
        reply=process_request(self.root,self.submit(),FixturePredictor())
        self.assertFalse(reply["model_inference_run"])
        with self.assertRaisesRegex(ValueError,"Synthetic fixture"):self.session.accept(reply,self.row,100.5)
        self.assertIsNone(self.bridge.plan)

    def test_expired_queued_request_skips_model_inference(self):
        path=self.submit();request=read_json(path);request["deadline_wall_unix_s"]=1.;write_json(path,request)
        class MustNotCall:
            def predict(self,request):raise AssertionError("Model should not be called")
        reply=process_request(self.root,path,MustNotCall())
        self.assertFalse(reply["model_inference_run"])
        self.assertIn("expired",reply["error"])

    def test_nonfinite_measured_state_cannot_be_allocated(self):
        self.row["yaw_rad"]=float("nan")
        with self.assertRaisesRegex(ValueError,"Non-finite measured"):self.bridge.tick(self.row,.05)

    def test_isaac_orchestrator_contains_no_pose_or_velocity_setters(self):
        tree=ast.parse((PROJECT/"scripts/navigation_experiment.py").read_text())
        forbidden={"set_world_pose","set_local_pose","set_transforms","set_velocities","set_linear_velocity","set_angular_velocity","set_state"}
        writes=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden]
        self.assertEqual(writes,[])


if __name__=="__main__":unittest.main(verbosity=2)
