"""Semantic navigation safety contracts; synthetic fixtures are not live runs."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image,ImageDraw

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))
from approach_planning import ObservedGrid
from depth_localization import DepthFrame
from gpu_lease import OffloadedPredictor
from online_terrain_map import OnlineTerrainMap
from semantic_agent import refine_structural_pixel,verify_flat_top_identity
from semantic_execution import SensedMediumSwitcher
from semantic_worker import process_request
from navigation_io import write_json


class SemanticContracts(unittest.TestCase):
    def test_unknown_barrier_only_permits_reachable_frontier_route(self):
        mapper=OnlineTerrainMap([10,10],extent_m=20.)
        mapper.state[:]=1;mapper.state[:,18:24]=-1;mapper.height=np.zeros(mapper.shape)
        mapper.safe=ObservedGrid(mapper.state,mapper.origin,mapper.resolution,mapper.footprint_radius).safe
        route=mapper.plan([2.,5.],[10.,5.])
        self.assertEqual(route['kind'],'observed_frontier_exploration')
        self.assertLess(max(x for x,y in route['path_xy_m']),4.5)
        self.assertTrue(mapper.segment_safe([2.,5.],route['local_goal_xy_m']))

    def test_support_evidence_does_not_fill_unobserved_space_between_poses(self):
        mapper=OnlineTerrainMap([0.,0.],extent_m=20.)
        row={'x_m':0.,'y_m':0.,'z_m':.22,'yaw_rad':0.,'left_normal_n':235.,'right_normal_n':235.,'buoyancy_n':0.}
        mapper.update_state(row);mapper.update_state({**row,'x_m':5.})
        self.assertEqual(mapper.state[mapper.cell([0.,0.])],1)
        self.assertEqual(mapper.state[mapper.cell([5.,0.])],1)
        self.assertEqual(mapper.state[mapper.cell([2.5,0.])],-1)

    def test_contact_buoyancy_switching_requires_no_scene_depth(self):
        robot={'mass_kg':48.,'gravity_m_s2':9.81};switch=SensedMediumSwitcher(robot)
        for i in range(1,50):
            mode=switch.update({'time_s':i*.05,'left_normal_n':0.,'right_normal_n':0.,'buoyancy_n':470.88},.05)
        self.assertEqual(mode,'WATER')
        for i in range(50,110):
            mode=switch.update({'time_s':i*.05,'left_normal_n':235.44,'right_normal_n':235.44,'buoyancy_n':0.},.05)
        self.assertEqual(mode,'LAND')

    def test_grounding_refinement_cannot_pick_a_structure_outside_the_vlm_box(self):
        with tempfile.TemporaryDirectory(prefix='l17_semantic_pixel_') as temp:
            path=Path(temp)/'image.png';image=Image.new('RGB',(32,32),(80,100,120))
            ImageDraw.Draw(image).rectangle((16,8,18,24),fill=(200,95,25));image.save(path)
            frame=types.SimpleNamespace(depth_m=np.full((32,32),3.))
            grounding={'visible':True,'format_valid':True,'bbox_xyxy':[12,5,25,27],'point_uv':[15,16]}
            result,mask=refine_structural_pixel(frame,path,grounding)
            self.assertEqual(result['point_uv'],[16,16]);self.assertTrue(mask[16,16])
            result,mask=refine_structural_pixel(frame,path,{**grounding,'bbox_xyxy':[0,0,10,10],'point_uv':[5,5]})
            self.assertFalse(result['visible']);self.assertIsNone(mask)

    def test_model_exception_still_returns_weights_to_cpu(self):
        events=[]
        class Model:
            device='cpu'
            def parameters(self):return [types.SimpleNamespace(numel=lambda:16,element_size=lambda:2)]
            def to(self,device):self.device=device;events.append(device);return self
        model=Model()
        cuda=types.SimpleNamespace(device=lambda device:nullcontext(),mem_get_info=lambda:(64*1024**3,64*1024**3),
                                   synchronize=lambda:events.append('synchronize'),empty_cache=lambda:events.append('empty'))
        def broken(request):
            self.assertEqual(model.device,'cuda:0');raise RuntimeError('backend inference failed')
        backend=types.SimpleNamespace(model=model,torch=types.SimpleNamespace(cuda=cuda),predict=broken)
        with patch('gpu_lease.own_gpu_lease',lambda project:nullcontext()):
            with self.assertRaisesRegex(RuntimeError,'backend inference failed'):OffloadedPredictor(PROJECT,backend,'cuda:0').predict({})
        self.assertEqual(model.device,'cpu');self.assertEqual(events[-2:],['synchronize','empty'])

    def test_foreign_semantic_worker_request_never_reaches_planner(self):
        with tempfile.TemporaryDirectory(prefix='l17_semantic_protocol_') as temp:
            root=Path(temp);path=root/'request1.request.json'
            write_json(path,{'schema':'L17.semantic_request.v1','request_id':'request1','worker_id':'other_worker',
                             'controller_epoch':'epoch','observation_sha256':'fixture'})
            class MustNotCall:
                def update(self,request):raise AssertionError('Foreign request reached semantic planner')
            reply=process_request(root,path,MustNotCall(),'own_worker')
            self.assertFalse(reply['ok']);self.assertIn('another worker',reply['error'])

    def test_pointed_apex_above_a_lower_deck_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='l17_top_identity_') as temp:
            root=Path(temp);image=root/'input.png';Image.new('RGB',(64,64),(160,80,25)).save(image)
            client=types.SimpleNamespace(describe=lambda *a,**k:json.dumps({'highest_top_shape':'pointed_apex','upper_structure_fully_visible':True}))
            result=verify_flat_top_identity(client,image,{'bbox_xyxy':[10,10,50,60]},root/'crop.png')
            self.assertFalse(result['accepted'])


if __name__=='__main__':unittest.main(verbosity=2)
