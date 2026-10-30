"""MSGNav decision module adapted to an external, observed scene graph.

This does not construct perception, frontiers or a complete MSGNav scene. Input
objects must come from recorded observations, not hidden Isaac Stage objects.
"""
import base64
import heapq
from io import BytesIO
import logging
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from navigation_contracts import finite_vector, isaac_world_to_habitat
from navigation_sources import load_functions

FUNCTIONS = ["encode_tensor2base64", "format_question", "Key_Subgraph_Selection",
             "Prompt_with_AVU_and_CLR", "Prompt_without_AVU", "format_exploreonly_prompt",
             "format_end_prompt", "format_prefiltering_prompt", "get_prefiltering_objs",
             "related_object_KSS", "edge_pruning_KSS", "explore_two_step", "task_check"]


def read_rgb(path, root):
    resolved = (Path(root) / path).resolve()
    if not resolved.is_file() or resolved.stat().st_size > 12_000_000:
        raise ValueError("Use an existing observation image under 12 MB")
    with Image.open(resolved) as image:
        return np.asarray(image.convert("RGB"))


def build_step(observation, image_root):
    if observation.get("source") not in ["perception", "synthetic_fixture"]:
        raise ValueError("Declare source=perception, or synthetic_fixture for interface tests")
    if not isinstance(observation.get("instruction"), str) or not observation["instruction"].strip():
        raise ValueError("An instruction is required")
    capture_time = float(observation["capture_time_s"])
    if not np.isfinite(capture_time) or capture_time < 0:
        raise ValueError("capture_time_s must be finite and nonnegative")
    pose = finite_vector(observation["anchor_pose_xyyaw"], 3, "anchor pose")
    objects = {}
    for obj in observation.get("objects", []):
        object_id = int(obj["id"])
        if object_id < 0 or object_id in objects:
            raise ValueError("Object IDs must be unique and nonnegative")
        center = isaac_world_to_habitat(obj["position_world_m"])
        if obj.get("approach_xy_m") is not None:
            finite_vector(obj["approach_xy_m"], 2, "observed approach point")
        objects[object_id] = {"id": object_id, "class_name": str(obj["class_name"]),
                              "bbox": SimpleNamespace(center=center), "room_label": str(obj.get("room_label", "unknown"))}
    snapshots = observation.get("snapshots", {})
    all_imgs = {str(key): read_rgb(value["image"], image_root) for key, value in snapshots.items()}
    for value in snapshots.values():
        if value.get("approach_xy_m") is not None:
            finite_vector(value["approach_xy_m"], 2, "image approach point")
    edges = {}
    image_to_edges = {key: [] for key in all_imgs}
    for edge in observation.get("edges", []):
        pair = tuple(int(x) for x in edge["object_ids"])
        if len(pair) != 2 or any(x not in objects for x in pair) or pair in edges:
            raise ValueError("Invalid or duplicate graph edge")
        images = list(dict.fromkeys(str(x) for x in edge["image_ids"]))
        if not images or any(key not in all_imgs for key in images):
            raise ValueError("Every graph edge needs an observed reference image")
        edges[pair] = SimpleNamespace(rel_img=images)
        for key in images:
            image_to_edges[key].append(pair)
    frontiers = observation.get("frontiers", [])
    for frontier in frontiers:
        finite_vector(frontier["approach_xy_m"], 2, "frontier approach point")
    views = [read_rgb(path, image_root) for path in observation.get("egocentric_images", [])]
    step = {"question": observation["instruction"], "task_type": "text",
            "objects": objects, "edges": edges, "all_imgs": all_imgs,
            "image_to_edges": image_to_edges,
            "frontier_imgs": [read_rgb(value["image"], image_root) for value in frontiers],
            "egocentric_views": views, "use_egocentric_views": bool(views),
            "prompt_h": 320, "prompt_w": 320,
            "CLR": observation.get("decision_history", {"cnt_step": 0, "max_step": 100})}
    return step, capture_time, pose


class MSGNavDecision:
    def __init__(self, client):
        self.client = client
        self.calls = []
        namespace = {"np": np, "Image": Image, "base64": base64, "BytesIO": BytesIO,
                     "heapq": heapq, "logging": logging, "call_openai_api": self._completion}
        load_functions("third_party/MSGNav/src/utils.py", ["resize_image"], namespace)
        self.functions = load_functions("third_party/MSGNav/src/explore_utils.py", FUNCTIONS, namespace)

    def _completion(self, system, contents):
        started = time.monotonic()
        result = self.client.msgnav_completion(system, contents)
        self.calls.append({"latency_s": time.monotonic() - started, "response": result,
                           "usage": getattr(self.client, "last_usage", None)})
        return result

    def decide(self, observation, image_root):
        step, capture_time, pose = build_step(observation, image_root)
        self.calls = []
        cfg = SimpleNamespace(prefiltering=True, top_k_categories=10, use_AVU=True,
                              use_room_filter=False, use_ollama=False)
        raw, image_map, reason, count = self.functions["explore_two_step"](step, cfg)
        if not isinstance(raw, str):
            raise RuntimeError("MSGNav produced no valid decision; do not substitute a guessed goal")
        match = re.fullmatch(r"(object|image|frontier) (\d+)(?:,\s*(.+))?", raw)
        if not match:
            raise RuntimeError("MSGNav returned an unsupported decision format")
        kind, id_text, category = match.groups()
        choice = int(id_text)
        approach, evidence = None, None
        if kind == "object":
            obj = next((value for value in observation.get("objects", []) if int(value["id"]) == choice), None)
            if obj is None:
                raise RuntimeError("MSGNav selected an unknown object")
            approach, evidence = obj.get("approach_xy_m"), choice
        elif kind == "frontier":
            if choice >= len(observation.get("frontiers", [])):
                raise RuntimeError("MSGNav selected an unknown frontier")
            approach = observation["frontiers"][choice]["approach_xy_m"]
            evidence = choice
        elif choice in image_map:
            evidence = str(image_map[choice])
            approach = observation["snapshots"][evidence].get("approach_xy_m")
        else:
            # Current camera selection requires a detector/depth back-projection.
            # A pixel or an image number is not a physical goal coordinate.
            evidence = {"egocentric_index": choice - len(image_map)}
        if approach is not None:
            approach = finite_vector(approach, 2, "selected approach point")
        return {"source": "MSGNav_original_decision_module", "observation_source": observation["source"],
                "capture_time_s": capture_time, "anchor_pose_xyyaw": pose,
                "choice_type": kind, "choice_id": choice, "category": category,
                "reason": reason, "raw_response": raw, "evidence": evidence,
                "goal_xy_m": approach, "goal_resolved": approach is not None,
                "task_success_verified": False, "compressed_snapshot_count": count,
                "api_calls": self.calls.copy(), "api_call_count": len(self.calls)}
