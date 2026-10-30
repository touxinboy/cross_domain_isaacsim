"""Build model inputs from real robot RGB-D bundles, without fabricating history."""
from collections import deque
import math
from pathlib import Path

import numpy as np

from depth_localization import DepthFrame
from navigation_contracts import finite_vector
from navigation_io import file_sha, own_path, read_json


class ObservationBuffer:
    def __init__(self, project, maxlen=32):
        self.project = Path(project).resolve()
        self.frames = deque(maxlen=maxlen)

    def append_capture(self, summary):
        if isinstance(summary, (str, Path)): summary = read_json(summary)
        cameras = {item["name"]: own_path(self.project, item["frame_file"]) for item in summary["frame_files"]}
        if set(cameras) != {"left", "front", "right"}: raise ValueError("Use exactly the robot's three calibrated views")
        frames = {key: DepthFrame.load(path) for key, path in cameras.items()}
        for frame in frames.values():
            if frame.metadata.get("camera_role") != "robot_mounted": raise ValueError("Fixed reference cameras are diagnostic only")
        front = frames["front"].metadata
        epoch = front.get("controller_epoch")
        capture = front.get("controller_time_s")
        if not isinstance(epoch, str) or not epoch: raise ValueError("Capture must record its robot-runtime clock epoch")
        if capture is None or not math.isfinite(float(capture)) or float(capture) < 0: raise ValueError("Missing finite controller-clock capture time")
        pose = finite_vector(front["robot_pose_xyyaw"], 3, "observed self pose")
        for frame in frames.values():
            meta = frame.metadata
            if meta.get("controller_epoch") != epoch or abs(float(meta["controller_time_s"]) - float(capture)) > .001:
                raise ValueError("Three views do not belong to the same paused controller state")
            if not np.allclose(meta["robot_pose_xyyaw"], pose, atol=1e-5): raise ValueError("Three views have inconsistent self poses")
        if self.frames:
            if self.frames[-1]["controller_epoch"] != epoch: raise ValueError("Clear observation history after a runtime reset")
            if float(capture) <= self.frames[-1]["capture_time_s"]: raise ValueError("Record a new physical time instead of duplicating a history frame")
        relative = lambda p: str(p.relative_to(self.project)).replace("\\", "/")
        item = {"capture_time_s": float(capture), "anchor_pose_xyyaw": pose, "controller_epoch": epoch,
                "observation_id": front["frame_id"], "tri_view": {}, "rgbd_frames": {}, "image_sha256": {}}
        item['mapping_rgbd_frames']=[]
        for entry in summary.get('mapping_frame_files',[]):
            path=own_path(self.project,entry['frame_file']);mapping=DepthFrame.load(path)
            if (mapping.metadata.get('camera_role')!='robot_mounted' or mapping.metadata.get('controller_epoch')!=epoch
                    or abs(float(mapping.metadata.get('controller_time_s',-1))-float(capture))>.001):
                raise ValueError('Mapping camera must share the paused robot runtime state')
            item['mapping_rgbd_frames'].append(relative(path))
        for key, path in cameras.items():
            meta = frames[key].metadata
            rgb = path.parent / meta["rgb_file"]
            item["tri_view"][key] = relative(rgb)
            item["rgbd_frames"][key] = relative(path)
            item["image_sha256"][key] = file_sha(rgb)
        self.frames.append(item)
        return item

    def observation(self, instruction):
        if len(self.frames) < 5: raise ValueError("Collect four earlier observations plus the current tri-view first")
        current = self.frames[-1]
        # Preserve a small recent temporal window. Every chosen frame has a
        # distinct measured simulation time, even if the robot stayed still.
        past = list(self.frames)[-13:-1]
        indices = np.linspace(0, len(past) - 1, 4, dtype=int)
        history = [{"image": past[i]["tri_view"]["front"], "xy_m": past[i]["anchor_pose_xyyaw"][:2],
                    "capture_time_s": past[i]["capture_time_s"], "image_sha256": past[i]["image_sha256"]["front"]}
                   for i in indices]
        return {"source": "perception", "instruction": instruction,
                "observation_id": current["observation_id"], "controller_epoch": current["controller_epoch"],
                "capture_time_s": current["capture_time_s"], "anchor_pose_xyyaw": current["anchor_pose_xyyaw"],
                "self_pose_source": "simulator_self_pose_baseline_not_visual_SLAM",
                "objects": [], "edges": [], "frontiers": [], "snapshots": {},
                "egocentric_images": [current["tri_view"][key] for key in ["left", "front", "right"]],
                "egocentric_rgbd_frames": [current["rgbd_frames"][key] for key in ["left", "front", "right"]],
                "mapping_rgbd_frames": current.get('mapping_rgbd_frames',[]),
                "pointgoal": {"history": history, "tri_view": current["tri_view"],
                              "tri_view_sha256": current["image_sha256"]}}
