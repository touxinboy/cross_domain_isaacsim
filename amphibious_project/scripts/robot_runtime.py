"""Own robot controller: native PhysX state/contact input and force/torque output."""

import csv
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import omni.kit.app
import omni.physics.tensors as tensors
import omni.physx
import omni.timeline
import omni.usd
from pxr import PhysicsSchemaTools

from amphibian_dynamics import AmphibianForces
from river_geometry import RiverGeometry

_RUNTIME = None


class WaterEnvironment:
    def __init__(self, kind, project):
        self.kind = kind
        self.default_ground_mu = .8
        self.field = None
        if kind == "river":
            self.config = json.loads((Path(project)/"configs/river_forest.json").read_text())
            self.field = RiverGeometry(self.config)

    def water_columns(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        if self.kind == "land":
            return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        if self.kind == "water":
            wet = (np.abs(x) < 16) & (np.abs(y) < 12)
            return np.zeros_like(x), np.full_like(x, -3.), wet.astype(float)
        heights = np.array([self.field.height(float(a), float(b)) for a,b in zip(x,y)])
        wet = (heights < self.field.water_z) & (np.abs(x) < self.field.half_x) & (np.abs(y) < self.field.half_y)
        return np.full_like(x, self.field.water_z), heights, wet.astype(float)

    def ground_mu(self, x, y, collider):
        if "Rock_" in collider or "Trunk" in collider:
            return .65
        if self.field:
            return self.config["materials"][self.field.soil(x,y)]["dynamic_friction"]
        return self.default_ground_mu


class RobotRuntime:
    def __init__(self, project, environment_kind):
        self.project = Path(project).resolve(strict=True)
        if not self.project.is_relative_to(Path("/home/ubuntu/sdc/L17").resolve(strict=True)):
            raise RuntimeError("Robot runtime must remain inside L17.")
        self.cfg = json.loads((self.project/"configs/robot_v1.json").read_text())
        self.environment = WaterEnvironment(environment_kind, self.project)
        self.model = AmphibianForces(self.cfg)
        self.stage = omni.usd.get_context().get_stage()
        if not Path(self.stage.GetRootLayer().realPath).resolve().is_relative_to(self.project):
            raise RuntimeError("Refusing to control a scene outside this project.")
        if not self.stage.GetPrimAtPath("/World/Robot"):
            raise RuntimeError("The own robot is absent from this scene.")
        self.commands = {"mode":"HYBRID","tracks":[0.,0.],"thrusters":[0.,0.]}
        self.buoyancy_enabled = True
        self.timeline = omni.timeline.get_timeline_interface()
        self.sim_view = None
        self.body_view = None
        self.sim_time = 0.0
        self.clock_epoch = uuid.uuid4().hex
        self.history = []
        self.latest = None
        self.error = None
        self.phase = "manual"
        self.initialization_attempts = 0
        self.last_dt = 1/self.cfg["physics_hz"]
        self.path_cache = {}
        self.pending = self.empty_contacts()
        self.step_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_step)
        self.contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.on_contacts)
        self.timeline_sub = self.timeline.get_timeline_event_stream().create_subscription_to_pop(self.on_timeline)

    def empty_contacts(self):
        return [{"normal_n":0.,"normal_weighted":np.zeros(3),"mu_weighted":0.} for _ in range(2)]

    def shutdown(self):
        self.zero()
        self.step_sub = None
        self.contact_sub = None
        self.timeline_sub = None
        if self.sim_view:
            self.sim_view.invalidate()
        self.body_view = None
        self.sim_view = None

    def on_timeline(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self.zero()
            self.clock_epoch = uuid.uuid4().hex
            self.body_view = None
            self.sim_view = None
            self.pending = self.empty_contacts()
            self.model.reset()
            self.initialization_attempts = 0

    def zero(self):
        self.commands["tracks"] = [0.,0.]
        self.commands["thrusters"] = [0.,0.]

    def set_commands(self, mode="HYBRID", tracks=None, thrusters=None):
        if mode not in ["LAND","WATER","HYBRID"]:
            raise ValueError("Unknown actuator mode.")
        self.commands["mode"] = mode
        if tracks is not None:
            self.commands["tracks"] = np.clip(np.asarray(tracks,dtype=float),-1,1).tolist()
        if thrusters is not None:
            self.commands["thrusters"] = np.clip(np.asarray(thrusters,dtype=float),-1,1).tolist()

    def decode_path(self, key):
        if key not in self.path_cache:
            self.path_cache[key] = str(PhysicsSchemaTools.intToSdfPath(key))
        return self.path_cache[key]

    def on_contacts(self, headers, data, *unused):
        if not self.timeline.is_playing() or omni.usd.get_context().get_stage() != self.stage:
            return
        try:
            for h in headers:
                p0, p1 = self.decode_path(h.collider0), self.decode_path(h.collider1)
                own, other = (p0,p1) if p0.startswith("/World/Robot/") else (p1,p0)
                if not own.startswith("/World/Robot/Tracks/"):
                    continue
                side = 0 if "/Tracks/Left/" in own else 1
                for index in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                    item = data[index]
                    normal = np.array(item.normal,dtype=float)
                    if normal[2] < 0:
                        normal = -normal
                    if normal[2] < .4:
                        continue
                    impulse = np.asarray(item.impulse,dtype=float)
                    magnitude = abs(float(impulse @ normal)) if impulse.ndim else abs(float(impulse))
                    load = magnitude/max(1e-6,self.last_dt)
                    point = np.array(item.position,dtype=float)
                    acc = self.pending[side]
                    acc["normal_n"] += load
                    acc["normal_weighted"] += normal*load
                    acc["mu_weighted"] += self.environment.ground_mu(float(point[0]),float(point[1]),other)*load
        except Exception:
            self.fail("contact_report",traceback.format_exc())

    def initialize_view(self):
        self.sim_view = tensors.create_simulation_view("numpy")
        self.sim_view.set_subspace_roots("/")
        self.body_view = self.sim_view.create_rigid_body_view("/World/Robot")
        if self.body_view.count != 1:
            self.body_view = None
            raise RuntimeError("Expected one dynamic robot rigid body.")

    def fail(self, phase, error):
        self.error = phase+": "+error
        (self.project/"results/robot_v1").mkdir(exist_ok=True)
        (self.project/"results/robot_v1/runtime_error.txt").write_text(self.error)
        print("L17_ROBOT_RUNTIME_ERROR "+self.error,flush=True)
        self.timeline.pause()

    def on_step(self, dt):
        if not self.timeline.is_playing() or omni.usd.get_context().get_stage() != self.stage or self.error:
            return
        self.last_dt = float(dt)
        if self.body_view is None:
            try:
                self.initialize_view()
            except Exception:
                self.initialization_attempts += 1
                if self.initialization_attempts > 5:
                    self.fail("tensor_initialization",traceback.format_exc())
                return
        try:
            transform = self.body_view.get_transforms()[0].copy()
            velocity = self.body_view.get_velocities()[0].copy()
            contacts = []
            for item in self.pending:
                n = item["normal_n"]
                contacts.append({"normal_n":n,"normal":(item["normal_weighted"]/n).tolist() if n>1e-6 else [0,0,1],
                                 "mu":item["mu_weighted"]/n if n>1e-6 else self.environment.default_ground_mu})
            self.pending = self.empty_contacts()
            force, torque, state = self.model.compute(transform[:3],transform[3:],velocity,self.commands,
                                                       contacts,self.environment,float(dt),self.buoyancy_enabled)
            if not np.isfinite(force).all() or not np.isfinite(torque).all():
                raise RuntimeError("Non-finite external force.")
            if np.linalg.norm(force)>20000 or np.linalg.norm(torque)>5000:
                raise RuntimeError("External force exceeded the prototype sanity bound.")
            self.body_view.apply_forces_and_torques_at_position(force.astype(np.float32).reshape(1,3),
                torque.astype(np.float32).reshape(1,3),transform[:3].reshape(1,3),np.array([0],dtype=np.uint32),True)
            self.sim_time += float(dt)
            state.update({"time_s":self.sim_time,"phase":self.phase,"x_m":float(transform[0]),"y_m":float(transform[1]),"z_m":float(transform[2]),
                          "qx":float(transform[3]),"qy":float(transform[4]),"qz":float(transform[5]),"qw":float(transform[6]),
                          "vx_m_s":float(velocity[0]),"vy_m_s":float(velocity[1]),"vz_m_s":float(velocity[2]),
                          "wx_rad_s":float(velocity[3]),"wy_rad_s":float(velocity[4]),"wz_rad_s":float(velocity[5]),
                          "mode":self.commands["mode"],"left_track_command_m_s":self.commands["tracks"][0],
                          "right_track_command_m_s":self.commands["tracks"][1],
                          "left_thruster_command":self.commands["thrusters"][0],"right_thruster_command":self.commands["thrusters"][1]})
            self.latest = state
            self.history.append(state)
        except Exception:
            self.fail("physics_step",traceback.format_exc())

    def save_csv(self, path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.project):
            raise RuntimeError("Telemetry destination must be inside this project.")
        if not self.history:
            raise RuntimeError("No physics samples were recorded.")
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(self.history[0]))
            writer.writeheader();writer.writerows(self.history)
        return str(path)

    def status(self):
        return {"error":self.error,"time_s":self.sim_time,"samples":len(self.history),
                "commands":self.commands,"state":self.latest,"buoyancy_enabled":self.buoyancy_enabled}


def attach_robot_runtime(project, environment_kind="river"):
    global _RUNTIME
    if _RUNTIME:
        _RUNTIME.shutdown()
    _RUNTIME=RobotRuntime(project,environment_kind)
    return _RUNTIME


def get_robot_runtime():
    return _RUNTIME


def detach_robot_runtime():
    global _RUNTIME
    if _RUNTIME:
        _RUNTIME.shutdown()
        _RUNTIME=None
