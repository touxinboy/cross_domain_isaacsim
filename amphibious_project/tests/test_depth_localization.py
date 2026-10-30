"""Geometry and failure gates without Isaac, GPU, hidden objects or model calls."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from depth_localization import DepthFrame,calibration,localize_surface,project,unproject
from ground_target_api import parse_grounding


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.k=np.array([[100.,0,5.],[0,100.,5.],[0,0,1.]])
        self.t=np.eye(4)
        self.meta={'schema':'L17.rgbd.v1','depth_unit':'m','optical_axes':'x_right_y_down_z_forward',
                   'height':11,'width':11,'intrinsics':self.k.tolist(),'world_from_optical':self.t.tolist(),
                   'render_time_s':2.,'pose_time_s':2.,'depth_type':'distance_to_image_plane',
                   'frame_id':'fixture','rgb_sha256':'SYNTHETIC_FRAME'}
        self.ground={'visible':True,'format_valid':True,'image_sha256':'SYNTHETIC_FRAME','point_uv':[5.,5.]}

    def frame(self,depth=None):
        return DepthFrame(self.meta,np.full((11,11),2.,dtype=np.float32) if depth is None else depth)

    def test_optical_axes_and_metric_scale(self):
        actual=unproject([[5,5],[6,5],[5,6]],[2,2,2],self.k,self.t)
        np.testing.assert_allclose(actual,[[0,0,2],[.02,0,2],[0,.02,2]])

    def test_axial_and_radial_depth_are_distinct(self):
        k=np.eye(3)
        axial=unproject([[1,0]],[2],k,self.t,'distance_to_image_plane')[0]
        radial=unproject([[1,0]],[2],k,self.t,'distance_to_camera')[0]
        np.testing.assert_allclose(axial,[2,0,2])
        self.assertAlmostEqual(np.linalg.norm(radial),2.)

    def test_rotated_translated_world_round_trip(self):
        t=np.array([[0,0,1,8],[1,0,0,-3],[0,1,0,.7],[0,0,0,1]],float)
        pixels=np.array([[1.,3.],[7.,9.]])
        depths=np.array([2.,5.])
        world=unproject(pixels,depths,self.k,t)
        uv,z=project(world,self.k,t)
        np.testing.assert_allclose(uv,pixels,atol=1e-10)
        np.testing.assert_allclose(z,depths,atol=1e-10)

    def test_flat_surface_is_not_automatically_a_navigation_goal(self):
        result=localize_surface(self.frame(),self.ground)
        self.assertTrue(result['accepted'])
        np.testing.assert_allclose(result['world_point_m'],[0,0,2])
        self.assertFalse(result['navigation_goal_valid'])
        self.assertIsNone(result['goal_xy_m'])

    def test_nan_seed_is_rejected_without_nearest_pixel_guess(self):
        depth=np.full((11,11),2.,dtype=float);depth[5,5]=np.nan
        result=localize_surface(self.frame(depth),self.ground)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'],'invalid_depth_at_grounded_pixel')

    def test_hollow_structure_uses_seed_surface_not_box_median(self):
        depth=np.full((11,11),18.,dtype=float);depth[:,5]=2.
        result=localize_surface(self.frame(depth),self.ground)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['depth_m'],2.)
        self.assertAlmostEqual(result['local_cluster_fraction'],1/7)

    def test_unsupported_single_pixel_is_rejected(self):
        depth=np.full((11,11),18.,dtype=float);depth[5,5]=2.
        self.assertFalse(localize_surface(self.frame(depth),self.ground)['accepted'])

    def test_old_image_identity_is_rejected(self):
        self.ground['image_sha256']='ANOTHER_FRAME'
        with self.assertRaisesRegex(ValueError,'another RGB frame'):
            localize_surface(self.frame(),self.ground)

    def test_pose_timestamp_skew_is_rejected(self):
        self.meta['pose_time_s']=2.2
        with self.assertRaisesRegex(ValueError,'not synchronized'): self.frame().validate()

    def test_reflection_is_not_a_valid_camera_rotation(self):
        self.t[1,1]=-1
        with self.assertRaises(ValueError):calibration(self.k,self.t)

    def test_grounding_formats_and_bounds(self):
        result=parse_grounding('{"visible":true,"bbox_2d":[100,100,900,900],"point_2d":[500,500]}',640,360)
        self.assertEqual(result['point_uv'],[320.,180.])
        with self.assertRaises(ValueError):parse_grounding('{"visible":true,"bbox_2d":[100,100,900,900],"point_2d":[1001,0]}',640,360)

    def test_unknown_depth_type_and_millimetres_are_rejected(self):
        self.meta['depth_unit']='mm'
        with self.assertRaises(ValueError):self.frame().validate()
        with self.assertRaises(ValueError):unproject([[5,5]],[2],self.k,self.t,'unspecified')


if __name__=='__main__':unittest.main(verbosity=2)
