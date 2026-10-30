"""Physical invariants and loading paths, using the delivered collider vertices."""
import ast
import copy
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))
from scm_terrain import SCMTerrain, SCMTrackForces
from ground_environment import FiveSurfaceEnvironment


def make_terrain(spacing=None):
    ground = json.loads((PROJECT/'configs/five_surfaces.json').read_text())
    settings = json.loads((PROJECT/'configs/scm.json').read_text())
    robot = json.loads((PROJECT/'configs/robot_v1.json').read_text())
    text = (PROJECT/'robots/tracked_twin_thruster_v1.usda').read_text()
    points = [ast.literal_eval(s) for s in re.findall(r'def Mesh "Collider".*?point3f\[\] points = (\[[^\n]+\])',text,re.S)]
    assert len(points) == 2
    if spacing:
        settings['grid_spacing_m'] = spacing
    return SCMTerrain(ground,settings,robot,points)


class SCMTests(unittest.TestCase):
    def test_virgin_pressure_area_and_no_air_force(self):
        terrain = make_terrain()
        f, t, s = terrain.compute([-4,0,.20],[0,0,0,1],np.zeros(6),[0,0],1/120)
        self.assertGreater(s['scm_contact_cells'], 500)
        self.assertGreater(s['scm_contact_area_m2'], .25)
        self.assertLess(s['scm_contact_area_m2'], .34)
        self.assertGreater(f[2], 0)
        self.assertAlmostEqual(f[0], 0.)
        self.assertAlmostEqual(t[0], 0., delta=.2)
        ids = terrain.active
        self.assertTrue(np.all(terrain.plastic[ids] >= 0))
        xy = terrain.origin+(np.column_stack((ids//terrain.shape[1],ids%terrain.shape[1]))+.5)*terrain.spacing
        np.testing.assert_allclose(terrain.plastic[ids]+terrain.elastic[ids],terrain.h0-terrain.heights(xy))
        air, _, s = terrain.compute([-4,0,.50],[0,0,0,1],np.zeros(6),[1,1],1/120)
        np.testing.assert_allclose(air,0.)
        self.assertEqual(s['scm_contact_cells'],0)

    def test_loading_unloading_reload_has_permanent_rut(self):
        terrain = make_terrain()
        terrain.compute([-4,0,.19],[0,0,0,1],np.zeros(6),[0,0],1/120)
        permanent = terrain.plastic.copy()
        maximum = permanent.max()
        self.assertGreater(maximum,.01)
        terrain.compute([-4,0,.5],[0,0,0,1],np.zeros(6),[0,0],1/120)
        np.testing.assert_allclose(terrain.plastic,permanent)
        self.assertEqual(float(terrain.elastic.max()),0.)
        self.assertEqual(float(terrain.pressure.max()),0.)
        for _ in range(20):
            terrain.compute([-4,0,.19],[0,0,0,1],np.zeros(6),[0,0],1/120)
        np.testing.assert_allclose(terrain.plastic,permanent,atol=1e-12)
        terrain.compute([-4,0,.18],[0,0,0,1],np.zeros(6),[0,0],1/120)
        self.assertGreater(terrain.plastic.max(),maximum)

    def test_belt_velocity_generates_traction_and_reverses(self):
        terrain = make_terrain()
        f,_,_ = terrain.compute([-4,0,.20],[0,0,0,1],np.zeros(6),[.5,.5],1/120)
        self.assertGreater(f[0],10.)
        for _ in range(30):
            f,_,_ = terrain.compute([-4,0,.20],[0,0,0,1],np.zeros(6),[-.5,-.5],1/120)
        self.assertLess(f[0],-10.)
        self.assertTrue(np.all(np.linalg.norm(terrain.shear[terrain.active],axis=1) <= .100001))

    def test_equal_belt_and_patch_speed_has_no_initial_shear(self):
        terrain = make_terrain()
        f,_,s = terrain.compute([-4,0,.20],[0,0,0,1],[.4,0,0,0,0,0],[.4,.4],1/120)
        self.assertAlmostEqual(f[0],0.)
        self.assertAlmostEqual(s['left_slip_ratio'],0.)

    def test_turning_produces_drive_and_resisting_torques(self):
        terrain = make_terrain()
        _,t,_ = terrain.compute([-4,0,.20],[0,0,0,1],np.zeros(6),[-.4,.4],1/120)
        self.assertGreater(t[2],1.)
        terrain.release_contacts()
        _,t,s = terrain.compute([-4,0,.20],[0,0,0,1],[0,0,0,0,0,.5],[0,0],1/120)
        self.assertLess(t[2],-1.)
        self.assertGreater(s['left_lateral_slip_rms_m_s'],.01)

    def test_grid_refinement_preserves_material_response(self):
        forces=[]
        for spacing in (.02,.01):
            terrain=make_terrain(spacing)
            f,_,_=terrain.compute([-4,0,.20],[0,0,0,1],np.zeros(6),[0,0],1/120)
            forces.append(f[2])
        self.assertLess(abs(forces[0]/forces[1]-1),.03)

    def test_pitch_and_roll_use_actual_envelope(self):
        terrain=make_terrain()
        a=.08
        _,t,s=terrain.compute([-4,0,.20],[math.sin(a/2),0,0,math.cos(a/2)],np.zeros(6),[0,0],1/120)
        self.assertGreater(s['right_normal_n'],s['left_normal_n'])
        self.assertLess(t[0],0.)

    def test_rigid_contact_inputs_cannot_duplicate_scm_traction(self):
        a,b=make_terrain(),make_terrain()
        env=FiveSurfaceEnvironment(a.ground)
        cmd={'mode':'LAND','tracks':[.5,.5],'thrusters':[0,0]}
        zero=[{'normal_n':0.,'normal':[0,0,1],'mu':0.} for _ in range(2)]
        loaded=[{'normal_n':1000.,'normal':[0,0,1],'mu':1.} for _ in range(2)]
        f0,t0,_=SCMTrackForces(a.robot,a).compute([-4,0,.20],[0,0,0,1],np.zeros(6),copy.deepcopy(cmd),zero,env,1/120)
        f1,t1,state=SCMTrackForces(b.robot,b).compute([-4,0,.20],[0,0,0,1],np.zeros(6),copy.deepcopy(cmd),loaded,env,1/120)
        np.testing.assert_allclose(f0,f1)
        np.testing.assert_allclose(t0,t1)
        self.assertEqual(state['native_track_normal_n'],2000.)

    def test_state_export_contains_persistent_and_elastic_fields(self):
        terrain=make_terrain()
        terrain.compute([-4,0,.20],[0,0,0,1],np.zeros(6),[.5,.5],1/120)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'soil.npz'
            terrain.save(path)
            with np.load(path) as saved:
                np.testing.assert_allclose(saved['plastic'],terrain.plastic.reshape(terrain.shape))
                self.assertIn('elastic',saved.files)
                self.assertIn('settings_json',saved.files)


if __name__ == '__main__':
    unittest.main(verbosity=2)
