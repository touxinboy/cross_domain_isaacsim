"""Physical invariants of the prototype model, independent of Isaac Sim."""

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"scripts"))
from amphibian_dynamics import AmphibianForces, quaternion_matrix_xyzw


class Environment:
    default_ground_mu=.8
    def __init__(self,water):self.water=water
    def water_columns(self,x,y):
        return np.zeros_like(x),np.full_like(x,-3 if self.water else 0),np.full_like(x,float(self.water))


class ForceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((PROJECT/"configs/robot_v1.json").read_text())
        self.model=AmphibianForces(self.cfg)
        self.water=Environment(True);self.land=Environment(False)
        self.air_contacts=[{"normal_n":0.,"normal":[0,0,1],"mu":.8} for _ in range(2)]

    def test_archimedes_equilibrium_and_volume_limits(self):
        f,t,v=self.model.buoyancy(np.array([0.,0.,self.model.equilibrium_z]),np.eye(3),self.water)
        self.assertAlmostEqual(v,self.cfg["mass_kg"]/1000,places=10)
        self.assertAlmostEqual(f[2],self.cfg["mass_kg"]*9.81,places=7)
        self.assertLess(np.linalg.norm(t),1e-9)
        dry=self.model.buoyancy(np.array([0.,0.,1.]),np.eye(3),self.water)
        full=self.model.buoyancy(np.array([0.,0.,-1.]),np.eye(3),self.water)
        self.assertEqual(dry[2],0.)
        self.assertAlmostEqual(full[2],self.model.max_volume,places=10)

    def test_buoyancy_restores_small_roll(self):
        angle=math.radians(5);q=[math.sin(angle/2),0,0,math.cos(angle/2)]
        f,t,v=self.model.buoyancy(np.array([0.,0.,self.model.equilibrium_z]),quaternion_matrix_xyzw(q),self.water)
        self.assertLess(t[0],0.)

    def test_equilibrium_metrics_are_json_serializable(self):
        expected=self.model.equilibrium_z
        text=json.dumps({"predicted_z_m":expected,"within_tolerance":abs(expected-.0354)<.025})
        self.assertTrue(json.loads(text)["within_tolerance"])

    def test_actuators_cannot_drive_in_air(self):
        cmd={"mode":"HYBRID","tracks":[1,1],"thrusters":[1,1]}
        f,t,row=self.model.compute([0,0,1],[0,0,0,1],np.zeros(6),cmd,self.air_contacts,self.water,.1)
        self.assertLess(np.linalg.norm(f),1e-10)
        self.assertEqual(row["left_thrust_n"],0.)
        self.assertEqual(row["left_traction_n"],0.)

    def test_traction_is_bounded_by_contact_load(self):
        contacts=[{"normal_n":100.,"normal":[0,0,1],"mu":.4} for _ in range(2)]
        cmd={"mode":"LAND","tracks":[1,1],"thrusters":[0,0]}
        f,t,row=self.model.compute([0,0,.22],[0,0,0,1],np.zeros(6),cmd,contacts,self.land,1.)
        self.assertGreater(f[0],0.)
        self.assertLessEqual(abs(f[0]),80.00001)

    def test_differential_thrust_yaw_sign(self):
        cmd={"mode":"WATER","tracks":[0,0],"thrusters":[.1,.8]}
        f,t,row=self.model.compute([0,0,self.model.equilibrium_z],[0,0,0,1],np.zeros(6),cmd,self.air_contacts,self.water,.3)
        self.assertGreater(f[0],0.)
        self.assertGreater(t[2],0.)

    def test_drag_removes_mechanical_energy(self):
        position=np.array([0.,0.,self.model.equilibrium_z]);velocity=np.array([.7,-.3,.1,.4,-.2,.3])
        buoy,bt,_=self.model.buoyancy(position,np.eye(3),self.water)
        f,t,row=self.model.compute(position,[0,0,0,1],velocity,{"mode":"WATER"},self.air_contacts,self.water,.01)
        power=float((f-buoy)@velocity[:3]+(t-bt)@velocity[3:])
        self.assertLess(power,0.)


if __name__=="__main__":unittest.main(verbosity=2)
