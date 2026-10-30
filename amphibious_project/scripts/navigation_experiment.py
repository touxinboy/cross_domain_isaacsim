"""Own Isaac 4.5 stop/look/predict/execute experiment; not yet live-validated.

Uses existing physical forces, contact-driven reference switching and a separate
OmniNav model process. No robot pose or velocity is written by this module.
"""
import asyncio
import csv
import math
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd

from approach_planning import ObservedGrid
from navigation_execution import PhysicalNavigationBridge, safe_polyline_prefix
from navigation_io import own_path, read_json, validate_goal, write_json, waypoint_code_identity
from navigation_loop import NavigationSession
from navigation_observations import ObservationBuffer
import robot_runtime


class NavigationRuntime(robot_runtime.RobotRuntime):
    def __init__(self, project, bridge, experiment):
        self.bridge, self.experiment = bridge, experiment
        self.next_control = 0.
        self.control_rows = []
        self.last_control = None
        self.collision_events = []
        self.enabled = False
        super().__init__(project, "river")
        self.clock_epoch = uuid.uuid4().hex

    def on_contacts(self, headers, data, *unused):
        super().on_contacts(headers, data, *unused)
        if not self.timeline.is_playing() or self.stage != omni.usd.get_context().get_stage(): return
        try:
            for header in headers:
                names = [self.decode_path(header.collider0), self.decode_path(header.collider1)]
                if not any(p.startswith("/World/Robot/") for p in names): continue
                other = next((p for p in names if not p.startswith("/World/Robot/")), "")
                impulse = sum(float(np.linalg.norm(data[i].impulse))
                              for i in range(header.contact_data_offset, header.contact_data_offset+header.num_contact_data))
                if impulse > .02 and ("/Obstacles/" in other or "/NavigationTargets/" in other or "/Trunk" in other):
                    if not self.collision_events:
                        self.collision_events.append({'event':'obstacle_contact','time_s':self.sim_time,
                            'collider_paths':names,'impulse_ns':impulse,
                            'robot_xy_m':[self.latest['x_m'],self.latest['y_m']] if self.latest else None})
                    self.experiment.reason = "obstacle_contact"; self.zero(); self.timeline.pause()
        except Exception: self.fail("navigation_contact", traceback.format_exc())

    def on_step(self, dt):
        if self.experiment.session and self.clock_epoch!=self.experiment.session.epoch:
            self.enabled=False;self.zero();self.experiment.reason="runtime_epoch_changed";self.timeline.pause();return
        self.phase = self.experiment.phase
        previous = self.sim_time
        super().on_step(dt)
        if self.error or self.sim_time == previous or not self.latest: return
        try:
            row = self.latest
            if abs(row["roll_deg"]) > 35 or abs(row["pitch_deg"]) > 38:
                self.experiment.reason = "attitude_limit"
            if math.hypot(row["vx_m_s"],row["vy_m_s"]) > 2.5:
                self.experiment.reason = "speed_limit"
            if self.experiment.reason:
                self.zero(); self.timeline.pause(); return
            if self.sim_time+1e-6 < self.next_control: return
            control_dt = 1/self.experiment.cfg["control_hz"] if self.last_control is None else self.sim_time-self.last_control
            self.last_control = self.sim_time
            self.next_control = self.sim_time+1/self.experiment.cfg["control_hz"]
            result = self.bridge.tick(row, control_dt)
            command = result["command"]
            if self.enabled: self.set_commands(**command)
            else: self.zero()
            self.control_rows.append({"time_s": self.sim_time, "x_m": row["x_m"], "y_m": row["y_m"],
                                      "status": result["status"], "mode": command["mode"],
                                      "applied": self.enabled, "left_track": command["tracks"][0],
                                      "right_track": command["tracks"][1], "left_thruster": command["thrusters"][0],
                                      "right_thruster": command["thrusters"][1]})
        except Exception: self.zero(); self.fail("navigation_control", traceback.format_exc())

    def save_csv(self, path):
        result = super().save_csv(path)
        write_json(Path(path).parent/'contact_events.json', self.collision_events)
        return result


class NavigationExperiment:
    def __init__(self, project, request, rig_factory=None):
        self.project = Path(project).resolve()
        self.request = request
        self.rig_factory = rig_factory
        if request.get("schema") != "L17.navigation_run.v1" or request.get("mode") not in ["shadow","execute"]:
            raise ValueError("Invalid navigation experiment request")
        deadline=float(request.get("start_deadline_wall_unix_s",0))
        if not math.isfinite(deadline) or deadline<time.time():raise ValueError("Start request expired; prepare a new task before starting the own GUI")
        self.cfg = read_json(self.project/"configs/navigation_closed_loop.json")
        self.goal = request["goal"]; self.goal_xy = validate_goal(self.goal)
        self.output = own_path(self.project, request["output_relative"])
        self.stop_file = own_path(self.project, request["stop_relative"])
        self.queue = own_path(self.project, self.cfg["worker_queue"])
        self.runtime = self.session = None
        self.future = None; self.reason = None; self.error = None
        self.phase = "prepared"; self.cycles = 0
        self.started_wall = time.monotonic()
        self.grid = None
        if self.goal.get("grid_file"):
            self.grid = ObservedGrid.load(own_path(self.project, self.goal["grid_file"]))
        self.buffer = ObservationBuffer(self.project)
        self.output_created = False
        self.cleanup_errors = []

    def status(self):
        return {"run_id": self.request["run_id"], "phase": self.phase, "mode": self.request["mode"],
                "sim_time_s": self.runtime.sim_time if self.runtime else 0., "cycles": self.cycles,
                "reason": self.reason, "error": self.error, "output": str(self.output)}

    def request_stop(self): write_json(self.stop_file, {"reason": "user_stop"})

    def start(self):
        if self.future and not self.future.done(): raise RuntimeError("Navigation is already running")
        self.future = asyncio.ensure_future(self.run())
        return self.status()

    def check(self):
        if self.stop_file.exists(): raise RuntimeError("user_stop")
        if self.reason: raise RuntimeError(self.reason)
        if self.runtime and self.runtime.error: raise RuntimeError(self.runtime.error)
        if self.runtime and self.runtime.stage != omni.usd.get_context().get_stage(): raise RuntimeError("own_scene_changed")
        if self.runtime and self.session and self.runtime.clock_epoch != self.session.epoch: raise RuntimeError("runtime_epoch_changed")
        if time.monotonic()-self.started_wall > self.cfg["max_wall_s"]: raise RuntimeError("wall_timeout")
        if self.runtime and self.runtime.sim_time > self.cfg["max_sim_s"]: raise RuntimeError("simulation_timeout")

    async def update(self):
        self.check()
        await omni.kit.app.get_app().next_update_async()

    async def pause_physics(self):
        runtime=self.runtime
        if runtime is None or runtime.stage!=omni.usd.get_context().get_stage():return
        runtime.enabled=False;runtime.zero()
        timeline=omni.timeline.get_timeline_interface()
        timeline.pause()
        # Kit applies timeline transitions on an app update. Reading is_playing
        # immediately after pause can still report True, causing a sensor API
        # to restore playback after an otherwise static capture.
        await omni.kit.app.get_app().next_update_async()
        if timeline.is_playing():raise RuntimeError("Timeline did not pause before sensor/model work")

    async def advance(self, seconds, drive=False):
        runtime = self.runtime
        runtime.enabled = drive; runtime.zero()
        started = runtime.sim_time
        target = started + seconds
        omni.timeline.get_timeline_interface().play()
        try:
            while runtime.sim_time < target:
                await self.update()
                if (drive and runtime.control_rows and runtime.control_rows[-1]["time_s"] > started
                        and runtime.control_rows[-1]["status"] not in ["tracking", "heading_alignment"]): break
        finally:
            await self.pause_physics()

    def worker_ready(self):
        if not (self.queue/"ready.json").is_file() or not (self.queue/"heartbeat.json").is_file():
            raise RuntimeError("Own waypoint worker is not ready; start it and wait for ready.json")
        ready = read_json(self.queue/"ready.json")
        heartbeat = read_json(self.queue/"heartbeat.json")
        if not ready.get("model_loaded") or heartbeat.get("worker_id") != ready["worker_id"]:
            raise RuntimeError("Dedicated waypoint worker is not ready")
        if ready.get("project")!=str(self.project) or ready.get("code_sha256")!=waypoint_code_identity(self.project):
            raise RuntimeError("Worker is using a different project or older code; restart the own waypoint worker")
        lock=read_json(self.project/"configs/omninav_checkpoint.lock.json")
        expected=next(item["sha256"] for item in lock["files"] if item["path"]=="config.json")
        if ready.get("checkpoint_config_sha256")!=expected:raise RuntimeError("Worker checkpoint configuration differs from the pinned model")
        stopped=self.queue/"stopped.json"
        if stopped.exists() and read_json(stopped).get("worker_id")==ready["worker_id"]:
            raise RuntimeError("Waypoint worker has stopped; restart it before sending another task")
        if not -1 <= time.time()-float(heartbeat["wall_unix_s"]) <= 5: raise RuntimeError("Waypoint worker heartbeat is stale")
        return ready

    async def capture(self, rig):
        self.phase = "capturing"; self.runtime.zero()
        await self.pause_physics()
        summary = await rig.capture_all(runtime=self.runtime)
        await self.pause_physics()
        return self.buffer.append_capture(summary)

    async def run(self):
        timeline = omni.timeline.get_timeline_interface()
        try:
            self.output.mkdir(parents=True, exist_ok=False)
            self.output_created = True
            write_json(self.output/"request.json", self.request)
            write_json(self.output/"worker_at_start.json", self.worker_ready())
            stage = omni.usd.get_context().get_stage()
            if not stage or not Path(stage.GetRootLayer().realPath).resolve().is_relative_to(self.project):
                raise RuntimeError("Open the own L17 robot scene before starting navigation")
            task=read_json(self.project/"configs/task.json")
            if Path(stage.GetRootLayer().realPath).resolve()!=own_path(self.project,task["scene_file"]):
                raise RuntimeError("Open the configured own task scene: "+task["scene_file"])
            existing = robot_runtime.get_robot_runtime()
            if existing and any(abs(x)>1e-8 for x in existing.commands["tracks"]+existing.commands["thrusters"]):
                raise RuntimeError("Zero the own manual actuator commands before starting")
            timeline.pause(); robot_runtime.detach_robot_runtime()
            from river_geometry import RiverGeometry
            robot = read_json(self.project/"configs/robot_v1.json")
            field = RiverGeometry(read_json(self.project/"configs/river_forest.json"))
            control = read_json(self.project/"configs/crossing_experiment.json")
            bridge = PhysicalNavigationBridge(robot, field, control, self.cfg["max_plan_age_s"], self.cfg["max_step_m"])
            self.runtime = NavigationRuntime(self.project, bridge, self)
            # Register the sole own body-force callback so the manual UI points
            # at this runtime rather than attaching a duplicate controller.
            robot_runtime._RUNTIME = self.runtime
            self.session = NavigationSession(self.project, self.runtime.clock_epoch, bridge, self.cfg["worker_timeout_s"])
            if self.rig_factory is None:
                import rgbd_sensor_rig
                rgbd_sensor_rig.dispose_rig()
                rig_source = self.project/"scripts/rgbd_sensor_rig.py"
                exec(compile(rig_source.read_text(),str(rig_source),"exec"),rgbd_sensor_rig.__dict__)
                rig = rgbd_sensor_rig.get_rig(self.project)
            else:rig=self.rig_factory(self.project)
            self.phase = "settling"
            await self.advance(self.cfg["settle_s"])
            for index in range(5):
                if index: await self.advance(self.cfg["history_spacing_s"])
                await self.capture(rig)
            while self.cycles < self.cfg["max_inference_cycles"]:
                self.check(); worker=self.worker_ready()
                current = self.buffer.frames[-1]
                distance = math.dist(current["anchor_pose_xyyaw"][:2],self.goal_xy)
                if distance <= self.cfg["goal_radius_m"] and math.hypot(self.runtime.latest["vx_m_s"],self.runtime.latest["vy_m_s"]) < self.cfg["goal_speed_m_s"]:
                    self.reason = "arrival_pose_reached_semantic_success_unverified"; break
                self.cycles += 1
                observation = self.buffer.observation(self.goal.get("instruction","Go to the supplied arrival pose."))
                observation_path = self.output/(f"observation_{self.cycles:03d}.json")
                write_json(observation_path, observation)
                self.session.submit(observation_path,observation,self.goal,self.queue,time.monotonic())
                self.phase = "waiting_for_model_paused_physics"
                paused_sim_time=float(self.runtime.sim_time)
                reply_path = self.session.pending["reply_path"]
                while not reply_path.exists():
                    stopped=self.queue/"stopped.json"
                    if stopped.exists() and read_json(stopped).get("worker_id")==worker["worker_id"]:
                        raise RuntimeError("waypoint_worker_exited_during_inference")
                    if self.session.check_timeout(time.monotonic()): raise RuntimeError("waypoint_worker_timeout")
                    await self.update()
                    if abs(self.runtime.sim_time-paused_sim_time)>1e-6:
                        raise RuntimeError("physics_advanced_during_model_wait")
                self.session.events.append({"event":"inference_physics_pause_verified",
                    "start_sim_s":paused_sim_time,"end_sim_s":float(self.runtime.sim_time),
                    "advanced_sim_s":float(self.runtime.sim_time)-paused_sim_time})
                reply = read_json(reply_path)
                write_json(self.output/(f"head_{self.cycles:03d}.json"),reply)
                plan = self.session.accept(reply,self.runtime.latest,time.monotonic())
                if self.grid:
                    polyline=plan.get('path_xy_m') or [observation["anchor_pose_xyyaw"][:2],plan["goal_xy_m"]]
                    if plan.get('path_length_m',0.)>.08:
                        prefix,kept=safe_polyline_prefix(polyline,self.grid.segment_is_safe)
                        if len(prefix)<2 or kept<.08:
                            self.session.cancel("head_segment_outside_observed_safe_space")
                            raise RuntimeError("head_segment_outside_observed_safe_space")
                        if len(prefix)<len(polyline):
                            plan=self.runtime.bridge.truncate_plan(prefix)
                if self.request["mode"] == "shadow":
                    proposal = bridge.tick(self.runtime.latest,1/self.cfg["control_hz"])
                    write_json(self.output/"shadow_command.json", {**proposal,"applied_to_robot":False})
                    self.reason = "shadow_interface_recorded"; break
                self.phase = "executing_short_waypoint"
                await self.advance(self.cfg["execution_slice_s"],drive=True)
                # Reach/expiry pauses execution. Clear the old plan before a new
                # capture or stationary history update can resume the timeline.
                bridge.clear_plan()
                await self.advance(self.cfg["history_spacing_s"])
                await self.capture(rig)
                write_json(self.output/"progress.json",self.status())
            if not self.reason: self.reason = "inference_cycle_limit"
        except Exception:
            self.error = traceback.format_exc()
            if not self.reason: self.reason = "experiment_stopped"
        finally:
            if self.runtime:
                self.runtime.zero(); self.runtime.enabled = False
                if self.runtime.stage == omni.usd.get_context().get_stage(): timeline.pause()
            # Zero demand and release the physics subscription even if the disk
            # is full or a telemetry export fails. Never replace an older run's
            # results when mkdir(exist_ok=False) rejected a reused output path.
            try:
                if self.session and self.session.pending:self.session.cancel(self.reason or "experiment_finished")
                if self.output_created:
                    if self.runtime:
                        if self.runtime.history:self.runtime.save_csv(self.output/"telemetry.csv")
                        if self.runtime.control_rows:
                            with (self.output/"control.csv").open("w",newline="") as handle:
                                writer=csv.DictWriter(handle,fieldnames=list(self.runtime.control_rows[0]))
                                writer.writeheader();writer.writerows(self.runtime.control_rows)
                    if self.session:write_json(self.output/"events.json",self.session.events)
            except Exception:
                self.cleanup_errors.append(traceback.format_exc())
            finally:
                if self.runtime and robot_runtime.get_robot_runtime() is self.runtime:
                    robot_runtime.detach_robot_runtime()
                    if self.runtime.stage == omni.usd.get_context().get_stage():
                        try:robot_runtime.attach_robot_runtime(self.project,"river")
                        except Exception:self.cleanup_errors.append(traceback.format_exc())
            self.phase="finished"
            if self.output_created:
                try:
                    write_json(self.output/"summary.json",{**self.status(),"task_success_verified":False,
                        "cleanup_errors":self.cleanup_errors,"pose_or_velocity_overwrite":False,
                        "capture_and_inference_pause_physics":True,"switch_policy":"reference_ground_truth_baseline",
                        "isaac_physics_samples":len(self.runtime.history) if self.runtime else 0})
                except Exception:print("L17_NAVIGATION_EXPORT_ERROR",traceback.format_exc(),flush=True)
            print("L17_NAVIGATION_FINISHED",self.reason,str(self.output),flush=True)
