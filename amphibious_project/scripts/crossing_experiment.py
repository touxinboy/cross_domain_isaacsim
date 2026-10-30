"""Isaac Sim 4.5 experiment. Import only inside our own full Kit app.

Initial pose is set once in the in-memory root layer, which is never saved.
An initial session-layer pose would mask PhysX USD writeback and freeze the image.
All subsequent robot motion is
integrated by PhysX from the existing force/contact model. No scene Save().
"""

import asyncio
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import carb
import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, Usd, UsdGeom

from crossing_control import CrossingController, route_preflight, crossing_checkpoint
from crossing_report import write_report
from river_geometry import RiverGeometry
from robot_assets import project_path, set_camera
from robot_runtime import RobotRuntime, detach_robot_runtime


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


async def wait_for_complete_png(path, timeout_s=15.):
    """The 4.5 viewport capture awaitable can finish before the async disk write."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 1000:
            with path.open("rb") as handle:
                signature = handle.read(8)
                handle.seek(-12, 2)
                tail = handle.read(12)
            if signature == b"\x89PNG\r\n\x1a\n" and tail == b"\x00\x00\x00\x00IEND\xaeB`\x82":
                return
        await omni.kit.app.get_app().next_update_async()
    raise RuntimeError("PNG did not finish writing: " + str(path))


class CrossingRuntime(RobotRuntime):
    def __init__(self, project, controller):
        self.controller = controller
        self.next_decision_time = 0.
        self.obstacle_contacts = {}
        self.hull_ground_impulse_ns = 0.
        self.capture_pending = None
        self.captured_stages = []
        super().__init__(project, "river")

    def on_contacts(self, headers, data, *unused):
        super().on_contacts(headers, data, *unused)
        if not self.timeline.is_playing() or self.stage != omni.usd.get_context().get_stage():
            return
        try:
            for h in headers:
                p0, p1 = self.decode_path(h.collider0), self.decode_path(h.collider1)
                if p0.startswith("/World/Robot/"):
                    own, other = p0, p1
                elif p1.startswith("/World/Robot/"):
                    own, other = p1, p0
                else:
                    continue
                impulse = sum(float(np.linalg.norm(np.asarray(data[i].impulse, dtype=float)))
                              for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data))
                if ("/Obstacles/Rock_" in other or "/Trunk" in other) and impulse > .001:
                    record = self.obstacle_contacts.setdefault(other, {"first_time_s": self.sim_time, "total_impulse_ns": 0.})
                    record["total_impulse_ns"] += impulse
                    if record["total_impulse_ns"] >= .02:
                        self.controller.finish("obstacle_contact")
                        self.zero()
                if "/Tracks/" not in own and "/Terrain/" in other:
                    self.hull_ground_impulse_ns += impulse
        except Exception:
            self.controller.finish("contact_diagnostic_error")
            self.fail("crossing_contact", traceback.format_exc())

    def on_step(self, dt):
        before = self.sim_time
        super().on_step(dt)
        if self.sim_time == before or self.error or not self.latest:
            return
        try:
            if self.sim_time + 1e-6 >= self.next_decision_time:
                self.next_decision_time = self.sim_time + 1. / self.controller.cfg["control_hz"]
                command = self.controller.tick(self.latest)
                self.phase = self.controller.phase
                self.set_commands(**command)
            if self.controller.reason:
                self.zero()
                self.timeline.pause()
            elif not self.capture_pending:
                checkpoint = crossing_checkpoint(self.latest, self.controller, self.cfg,
                                                 self.controller.field, self.captured_stages)
                if checkpoint:
                    self.capture_pending = checkpoint
                    self.timeline.pause()
        except Exception:
            self.controller.finish("control_exception")
            self.zero()
            self.fail("crossing_controller", traceback.format_exc())


class CrossingExperiment:
    def __init__(self, project, request):
        self.project = Path(project).resolve(strict=True)
        self.request = request
        self.output = project_path(self.project, request["output_relative"])
        self.stop_file = project_path(self.project, request["stop_relative"])
        self.cfg = json.loads((self.project / "configs/crossing_experiment.json").read_text())
        self.robot = json.loads((self.project / "configs/robot_v1.json").read_text())
        self.scene_cfg = json.loads((self.project / "configs/river_forest.json").read_text())
        self.field = RiverGeometry(self.scene_cfg)
        self.runtime = self.controller = None
        self.error = None
        self.checkpoint_photos = {}
        self.started = datetime.now().astimezone().isoformat()
        self.sources = {p: hashlib.sha256((self.project / p).read_bytes()).hexdigest() for p in [
            self.cfg["scene"], "scenes/river_forest_v1.usda", "robots/tracked_twin_thruster_v1.usda",
            "scripts/robot_runtime.py", "scripts/amphibian_dynamics.py", "scripts/crossing_control.py",
            "scripts/crossing_experiment.py", "configs/robot_v1.json", "configs/river_forest.json",
            "configs/crossing_experiment.json"]}

    def progress(self, state):
        r = self.runtime
        write_json(self.output / "progress.json", {"state": state, "run_id": self.request["run_id"],
                   "phase": self.controller.phase if self.controller else "opening",
                   "sim_time_s": r.sim_time if r else 0., "samples": len(r.history) if r else 0,
                   "latest": r.latest if r else None, "error": self.error,
                   "checkpoint_photos": self.checkpoint_photos,
                   "updated": datetime.now().astimezone().isoformat()})

    async def open_scene(self, route):
        app = omni.kit.app.get_app()
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        detach_robot_runtime()
        for _ in range(3):
            await app.next_update_async()
        scene = project_path(self.project, self.cfg["scene"])
        ok, error = await omni.usd.get_context().open_stage_async(str(scene))
        if not ok:
            raise RuntimeError(error)
        stage = omni.usd.get_context().get_stage()
        start, goal = route["start_xy_m"], route["goal_xy_m"]
        x, y = start
        # Initial resting clearance must cover the complete track footprint.
        z = max(self.field.mesh_height(x + dx, y + dy) for dx in [-.54, 0., .54] for dy in [-.44, 0., .44]) + .25
        yaw = math.atan2(goal[1] - y, goal[0] - x)
        # Keep the initial pose in the same in-memory layer used by PhysX writeback.
        # The scene file is never saved; closing the stage discards these changes.
        with Usd.EditContext(stage, stage.GetRootLayer()):
            robot = stage.GetPrimAtPath("/World/Robot")
            if not robot:
                raise RuntimeError("Own robot asset is absent")
            robot.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))
            robot.GetAttribute("xformOp:orient").Set(Gf.Quatf(math.cos(yaw / 2), Gf.Vec3f(0, 0, math.sin(yaw / 2))))
            for name, xy in [("Start", start), ("Goal", goal)]:
                marker = stage.GetPrimAtPath("/World/TaskMarkers/" + name)
                if marker:
                    marker.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*xy, self.field.mesh_height(*xy)))
                    if name == "Start":
                        UsdGeom.Imageable(marker).MakeInvisible()
            # Use a fixed side view to show both banks throughout the full crossing.
            set_camera(stage, "/World/Cameras/CrossingTest", (1, y - 22, 17), (0, y, -.4), 28)
        get_active_viewport().set_active_camera("/World/Cameras/CrossingTest")
        settings = carb.settings.get_settings()
        settings.set_bool("/physics/updateToUsd", True)
        settings.set_int("/persistent/app/viewport/displayOptions", 1 << 10)
        for _ in range(5):
            await app.next_update_async()
        return stage

    async def capture_checkpoint(self):
        runtime = self.runtime
        name = runtime.capture_pending
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        record = {"stage": name, "telemetry": dict(runtime.latest),
                  "sim_time_s": runtime.sim_time, "pause_for_photo_only": True,
                  "captured": datetime.now().astimezone().isoformat()}
        try:
            pose = runtime.body_view.get_transforms()[0].copy()
            velocity = runtime.body_view.get_velocities()[0].copy()
            record["native_pose_xyzw"] = [float(v) for v in pose]
            record["native_velocity_world"] = [float(v) for v in velocity]
            x, y, z = map(float, pose[:3])
            stage = runtime.stage
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                set_camera(stage, "/World/Cameras/CrossingPhoto", (x - 1.8, y - 3.2, z + 1.65),
                           (x + .12, y, z + .02), 32)
            viewport = get_active_viewport()
            viewport.set_active_camera("/World/Cameras/CrossingPhoto")
            for _ in range(28):
                await omni.kit.app.get_app().next_update_async()
            composed = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Robot")).ExtractTranslation()
            record["usd_composed_position_m"] = [float(v) for v in composed]
            record["render_physics_position_delta_m"] = float(np.linalg.norm(np.asarray(composed) - pose[:3]))
            if record["render_physics_position_delta_m"] > .05:
                raise RuntimeError("Rendered USD robot pose does not match native PhysX pose")
            path = self.output / (name + ".png")
            job = capture_viewport_to_file(viewport, str(path))
            await job.wait_for_result()
            await wait_for_complete_png(path)
            record.update({"file": path.name, "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            print("L17_CROSSING_PHOTO " + json.dumps({"stage": name, "file": str(path),
                  "sim_time_s": runtime.sim_time, "position_m": pose[:3].tolist()}), flush=True)
        except Exception:
            record["error"] = traceback.format_exc()
        finally:
            self.checkpoint_photos[name] = record
            write_json(self.output / "checkpoint_photos.json", self.checkpoint_photos)
            runtime.captured_stages.append(name)
            runtime.capture_pending = None
            get_active_viewport().set_active_camera("/World/Cameras/CrossingTest")
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()
            if not self.controller.reason:
                timeline.play()

    async def run(self):
        app = omni.kit.app.get_app()
        timeline = omni.timeline.get_timeline_interface()
        reason = "startup_error"
        self.progress("opening")
        try:
            route = route_preflight(self.cfg, self.scene_cfg, self.robot, self.field)
            write_json(self.output / "route_preflight.json", route)
            write_csv(self.output / "route_profile.csv", route["profile"])
            if not route["passed"]:
                raise RuntimeError("Course/config preflight failed; see route_preflight.json")
            await self.open_scene(route)
            self.controller = CrossingController(self.cfg, self.robot, self.field, route["start_xy_m"],
                                                 route["goal_xy_m"], route["waypoints_xy_m"])
            self.runtime = CrossingRuntime(self.project, self.controller)
            self.runtime.phase = "settle"
            timeline.set_current_time(0.)
            timeline.play()
            wall_start = last_sample_wall = time.monotonic()
            next_progress = wall_start
            last_time = 0.
            while not self.controller.reason:
                await app.next_update_async()
                now = time.monotonic()
                if self.runtime.capture_pending:
                    await self.capture_checkpoint()
                    last_sample_wall = time.monotonic()
                    self.progress("running")
                    continue
                if self.runtime.error:
                    self.error = self.runtime.error
                    self.controller.finish("physics_runtime_error")
                elif self.stop_file.exists():
                    self.controller.finish("user_stop_requested")
                elif now - wall_start > self.cfg["wall_timeout_s"]:
                    self.controller.finish("wall_timeout")
                elif self.runtime.sim_time > last_time:
                    last_time, last_sample_wall = self.runtime.sim_time, now
                elif now - last_sample_wall > 30.:
                    self.controller.finish("physics_not_advancing")
                elif not timeline.is_playing():
                    self.controller.finish("user_paused_or_stopped")
                if now >= next_progress:
                    self.progress("running")
                    next_progress = now + 1.
            reason = self.controller.reason
        except Exception:
            self.error = traceback.format_exc()
            print("L17_CROSSING_ERROR\n" + self.error, flush=True)
        finally:
            if self.runtime:
                self.runtime.zero()
            timeline.pause()  # Retain the measured final pose for inspection.
            if self.runtime and self.runtime.history:
                self.runtime.save_csv(self.output / "physics.csv")
                write_csv(self.output / "commands.csv", self.controller.decisions)
            summary = self.controller.monitor.summary(self.runtime.latest, reason, "reference_physical_crossing") if self.controller and self.runtime else {
                "kind": "reference_physical_crossing", "result": reason, "crossing_passed": False, "checks": {}}
            if self.runtime:
                contacts = self.runtime.obstacle_contacts
                summary["obstacle_contacts"] = contacts
                summary["hull_ground_impulse_ns"] = self.runtime.hull_ground_impulse_ns
                no_obstacles = not any(v["total_impulse_ns"] >= .02 for v in contacts.values())
                summary["checks"]["no_obstacle_contact"] = no_obstacles
                decisions = self.controller.decisions
                gaps = [r["control_dt_s"] for r in decisions[1:]]
                timing_ok = bool(gaps and max(gaps) <= 1. / self.cfg["control_hz"] + 2. / self.cfg["physics_hz"])
                summary["checks"]["control_rate_met"] = timing_ok
                summary["control_interval_mean_s"] = float(np.mean(gaps)) if gaps else None
                summary["control_interval_max_s"] = max(gaps) if gaps else None
                summary["crossing_passed"] = bool(all(summary["checks"].values()))
            summary.update({"run_id": self.request["run_id"], "started": self.started,
                            "finished": datetime.now().astimezone().isoformat(), "error": self.error,
                            "source_sha256": self.sources, "physics_hz": self.cfg["physics_hz"],
                            "control_hz": self.cfg["control_hz"], "parameters_calibrated": False,
                            "checkpoint_photos": self.checkpoint_photos,
                            "three_checkpoint_photos_saved": all(name in self.checkpoint_photos and "file" in self.checkpoint_photos[name]
                                for name in ["01_entering_water", "02_swimming", "03_landing"]),
                            "navigation_model": "handwritten_waypoint_tracker_and_contact_buoyancy_switcher",
                            "telemetry_source": "native_PhysX_ground_truth",
                            "pose_overwrite_during_run": False,
                            "command_timing": "commands.csv uses observed time; new commands apply from the next physics step (<=1/120 s delay)",
                            "simulation_claim": "Only a completed server run can establish full crossing performance."})
            if self.error:
                summary["crossing_passed"] = False
            write_json(self.output / "summary.json", summary)
            try:
                if self.runtime and self.runtime.history:
                    write_report(self.output)
                    for _ in range(15):
                        await app.next_update_async()
                    job = capture_viewport_to_file(get_active_viewport(), str(self.output / "final_view.png"))
                    await job.wait_for_result()
                    await wait_for_complete_png(self.output / "final_view.png")
            except Exception:
                (self.output / "optional_visual_export_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self.progress("passed" if summary["crossing_passed"] else "failed")
            print("L17_CROSSING_RESULT " + json.dumps(summary, allow_nan=False), flush=True)
            if self.request.get("close_when_done"):
                if self.runtime:
                    self.runtime.shutdown()
                # Discard the in-memory stage; never save a transient test pose/camera.
                await omni.usd.get_context().close_stage_async()
                # Closes only this newly launched own app without a save prompt.
                app.post_uncancellable_quit(0 if summary["crossing_passed"] else 2)
        return summary
