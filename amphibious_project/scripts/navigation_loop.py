"""Pure request/result/control coordinator, shared with the offline replay.

The worker is a separate process. Request/epoch/image identities and both
simulation-age and wall-time bounds are checked before actuator allocation.
"""
import math
from pathlib import Path
import time
import uuid

import numpy as np

from navigation_io import file_sha, own_path, validate_goal, write_json


class NavigationSession:
    def __init__(self, project, epoch, bridge, worker_timeout_s=120, allow_test_results=False):
        self.project = Path(project).resolve()
        self.epoch = epoch
        self.bridge = bridge
        if not math.isfinite(worker_timeout_s) or worker_timeout_s <= 0: raise ValueError("Invalid worker timeout")
        self.timeout = worker_timeout_s
        self.pending = None
        self.allow_test_results = allow_test_results
        self.events = []

    def submit(self, observation_path, observation, goal, queue, wall_now):
        if self.pending is not None: raise RuntimeError("One inference request is already outstanding")
        validate_goal(goal)
        if observation.get("controller_epoch") != self.epoch: raise ValueError("Observation belongs to another runtime epoch")
        path = own_path(self.project, observation_path)
        queue = own_path(self.project, queue)
        relative = lambda p: str(p.relative_to(self.project)).replace("\\", "/")
        request = {"schema": "L17.waypoint_request.v1", "request_id": uuid.uuid4().hex,
                   "controller_epoch": self.epoch, "observation_file": relative(path),
                   "observation_sha256": file_sha(path), "image_root": ".", "goal": goal,
                   "deadline_wall_unix_s":time.time()+self.timeout}
        request_path = queue / (request["request_id"] + ".request.json")
        write_json(request_path, request)
        self.pending = {"request": request, "observation": observation, "started_wall_s": float(wall_now),
                        "reply_path": queue / (request["request_id"] + ".result.json")}
        self.events.append({"event": "inference_submitted", "request_id": request["request_id"],
                            "capture_time_s": observation["capture_time_s"]})
        return request_path

    def cancel(self, reason):
        if self.pending is not None:
            reply=self.pending["reply_path"]
            marker=reply.with_name(reply.name.removesuffix(".result.json")+".cancelled.json")
            write_json(marker,{"reason":reason})
        self.pending = None
        self.bridge.clear_plan()
        self.events.append({"event": "plan_cleared", "reason": reason})

    def accept(self, reply, state, wall_now):
        try:
            if self.pending is None: raise ValueError("No pending request for this result")
            pending = self.pending
            request, observation = pending["request"], pending["observation"]
            elapsed = float(wall_now) - pending["started_wall_s"]
            if not 0 <= elapsed <= self.timeout: raise ValueError("Waypoint worker exceeded its wall-time budget")
            if reply.get("schema") != "L17.waypoint_result.v1": raise ValueError("Invalid worker result schema")
            for key in ["request_id", "controller_epoch", "observation_sha256"]:
                if reply.get(key) != request[key]: raise ValueError("Worker result identity mismatch: " + key)
            if not reply.get("ok"): raise RuntimeError("Waypoint inference failed: " + str(reply.get("error")))
            result = reply["result"]
            if result.get("test_fixture_only") and not self.allow_test_results:
                raise ValueError("Synthetic fixture results cannot drive a live experiment")
            if abs(float(result["capture_time_s"]) - float(observation["capture_time_s"])) > 1e-6:
                raise ValueError("Waypoint result changed the observation time")
            if not np.allclose(result["anchor_pose_xyyaw"], observation["anchor_pose_xyyaw"], atol=1e-8):
                raise ValueError("Waypoint result changed the observation anchor")
            plan = self.bridge.accept_head(result, state["time_s"],
                                           current_pose=[state[k] for k in ["x_m", "y_m", "yaw_rad"]])
            self.pending = None
            self.events.append({"event": "head_accepted", "request_id": request["request_id"],
                                "inference_wall_s": elapsed, "plan": plan})
            return plan
        except Exception:
            self.cancel("worker_result_rejected")
            raise

    def check_timeout(self, wall_now):
        if self.pending and float(wall_now) - self.pending["started_wall_s"] > self.timeout:
            self.cancel("worker_timeout")
            return True
        return False
