"""Physical geometry and velocity checks for the circulating band approximation."""
import json
import math
import unittest
import numpy as np
from test_scm import make_terrain, PROJECT
from track_motion import CirculatingTracks
from amphibian_dynamics import quaternion_matrix_xyzw


def motion():
    robot=json.loads((PROJECT/'configs/robot_v1.json').read_text())
    settings=json.loads((PROJECT/'configs/tracks.json').read_text())
    return CirculatingTracks(robot,settings)


class TrackTests(unittest.TestCase):
    def test_loop_closes_and_speed_reverses(self):
        track=motion()
        for s in (0.,.3,.9,1.4,2.):
            a=track.path(np.array([s,s+track.perimeter]))
            for value in a:
                np.testing.assert_allclose(value[0],value[1],atol=1e-12)
        before=track.poses(0)[0].copy()
        track.advance([.4,-.4],.001)
        after=track.poses(0)[0]
        bottom=np.where(before[:,2]<-.2049)[0][2]
        self.assertAlmostEqual((after[bottom,0]-before[bottom,0])/.001,-.4)
        self.assertGreater(track.phase[1],0.)
        track.advance([-.4,.4],.001)
        np.testing.assert_allclose(track.poses(0)[0],before,atol=1e-12)
        track.reset()
        np.testing.assert_array_equal(track.distance,[0.,0.])

    def test_plate_and_grouser_are_different_contact_heights(self):
        track=motion()
        center=track.poses(0)[0][5]
        points=np.array([[center[0],.35,-.3],[center[0]+.015,.35,-.3]])
        height,_,lug,_=track.ray_hits(0,points,[0,0,1],points)
        np.testing.assert_allclose(points[:,2]+height,[-.22,-.212],atol=1e-12)
        np.testing.assert_array_equal(lug,[True,False])

    def test_broad_phase_matches_exhaustive_shoe_intersection(self):
        track=motion()
        terrain=make_terrain()
        # Rays cover the full vehicle projection. Moderate slopes exceed those
        # permitted by the route controller, and are tested at several phases.
        xy=np.stack(np.meshgrid(np.linspace(-.6,.6,81),np.linspace(.18,.50,23)),axis=-1).reshape(-1,2)
        origin_world=np.column_stack((xy,np.zeros(len(xy))))
        for angles in ((0.,0.,0.),(.2,.15,.6),(-.2,-.15,-.5)):
            roll,pitch,yaw=angles
            cr,sr,cp,sp,cy,sy=math.cos(roll/2),math.sin(roll/2),math.cos(pitch/2),math.sin(pitch/2),math.cos(yaw/2),math.sin(yaw/2)
            q=[sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy]
            rotation=quaternion_matrix_xyzw(q)
            origin=(origin_world-[0,0,.20])@rotation
            for phase in (0.,.013,.032):
                track.distance[0]=phase
                lo,valid=terrain.envelope_rays(0,origin,rotation[2,:],padding=.005)
                guess=origin+np.where(valid,lo,0.)[:,None]*rotation[2,:]
                near=track.ray_hits(0,origin,rotation[2,:],guess)[0]
                full=track.ray_hits(0,origin,rotation[2,:],origin,all_shoes=True)[0]
                # Hits above the original soil plane cannot generate force;
                # an upper run visible through a lower gap can be omitted.
                relevant=full<0
                np.testing.assert_allclose(near[relevant],full[relevant],atol=1e-10)

    def test_contact_motion_matches_material_point_finite_difference(self):
        track=motion()
        speed=.43
        track.set_speeds([speed,speed])
        # Test a straight shoe and two curved shoes, including angular motion.
        centers,angles,_=track.poses(0)
        for index in (4,19,44):
            local=np.array([.006,.02,-.011])
            def material_point():
                c,t,_=track.poses(0)
                co,si=np.cos(t[index]),np.sin(t[index])
                return c[index]+[co*local[0]-si*local[2],local[1],si*local[0]+co*local[2]]
            p0=material_point()
            # The analytical velocity at the same material point.
            c,t,k=track.poses(0)
            analytical=np.array([-speed*math.cos(t[index]),0.,-speed*math.sin(t[index])])+np.cross([0.,speed*k[index],0.],p0-c[index])
            track.advance([speed,speed],1e-7)
            p1=material_point()
            np.testing.assert_allclose((p1-p0)/1e-7,analytical,atol=1e-6)
            track.advance([-speed,-speed],1e-7)

    def test_moving_geometry_drives_soil_without_double_belt_subtraction(self):
        terrain=make_terrain()
        track=motion()
        terrain.track_surface=track
        track.set_speeds([.4,.4])
        f,_,s=terrain.compute([-4,0,.212],[0,0,0,1],[.4,0,0,0,0,0],[.4,.4],1/120)
        # Curved shoes include angular velocity; their outer tips do not have
        # exactly the pitch-line speed. The bottom straight run has zero slip.
        self.assertLess(abs(f[0]),2.)
        self.assertGreater(s['left_grouser_contact_cells'],100)
        self.assertLess(abs(s['left_slip_ratio']),.02)
        terrain.release_contacts()
        f,_,_=terrain.compute([-4,0,.212],[0,0,0,1],np.zeros(6),[.4,.4],1/120)
        self.assertGreater(f[0],5.)
        plastic=terrain.plastic.copy()
        terrain.compute([-4,0,.5],[0,0,0,1],np.zeros(6),[0,0],1/120)
        np.testing.assert_array_equal(terrain.plastic,plastic)
        self.assertEqual(float(terrain.elastic.max()),0.)


if __name__=='__main__':
    unittest.main(verbosity=2)
