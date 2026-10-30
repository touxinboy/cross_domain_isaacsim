"""Dedicated OmniNav Slowfast inference, isolated from Isaac and Habitat.

No weights are downloaded automatically. This uses the author's pinned source
and local trained checkpoint, not a generic Qwen chat completion. Importing
does not import PyTorch/Transformers or allocate a GPU.
"""
import inspect
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from msgnav_bridge import read_rgb
from navigation_contracts import finite_vector, native_head_first_step, pointgoal_positions
from navigation_sources import PROJECT, load_functions, load_string_assignment
from navigation_io import file_sha

UPSTREAM = "third_party/OmniNav/infer_ovon_slowfast/qwen_utils.py"
IMAGE_FUNCTIONS = ["process_vision", "rescale_image_with_long_edge_and_random_scale",
                   "rescale_image_to_fixed_size", "crop_resize_image_magic_resolution",
                   "round_by_factor", "floor_by_factor", "ceil_by_factor", "smart_resize", "preprocess"]


def prepare_request(observation, target_xy_m, image_root):
    """Require four actual earlier observations; never fabricate history frames."""
    captured = float(observation["capture_time_s"])
    if not math.isfinite(captured) or captured < 0:
        raise ValueError("Invalid capture time")
    anchor = finite_vector(observation["anchor_pose_xyyaw"], 3, "anchor pose")
    pointgoal = observation["pointgoal"]
    history = pointgoal["history"]
    if len(history) != 4:
        raise ValueError("Record four real history frames before calling OmniNav")
    times = [float(frame["capture_time_s"]) for frame in history] + [captured]
    if not all(math.isfinite(t) and t >= 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("History capture times must be finite and strictly increasing")
    images = [read_rgb(frame["image"], image_root) for frame in history]
    images += [read_rgb(pointgoal["tri_view"][key], image_root) for key in ["left", "front", "right"]]
    for frame in history:
        if frame.get("image_sha256") and file_sha(Path(image_root)/frame["image"]) != frame["image_sha256"]:
            raise ValueError("A recorded history image was modified")
    for key, expected in pointgoal.get("tri_view_sha256", {}).items():
        if key not in ["left", "front", "right"] or file_sha(Path(image_root)/pointgoal["tri_view"][key]) != expected:
            raise ValueError("A current robot view was modified")
    positions = pointgoal_positions([frame["xy_m"] for frame in history], anchor[:2], target_xy_m, anchor)
    prompt = load_string_assignment(UPSTREAM, "get_result_fast", "content_text")
    return {"capture_time_s": captured, "anchor_pose_xyyaw": anchor, "images": images,
            "input_positions_scaled": positions, "content_text": prompt}


class OmniNavPointGoal:
    def __init__(self, checkpoint, device="cuda:0", attention="eager",load_on_cpu=False):
        self.checkpoint = Path(checkpoint).resolve()
        config_path = self.checkpoint / "config.json"
        if not config_path.is_file():
            raise ValueError("A local OmniNav_Slowfast checkpoint directory is required")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("waypoint_number") != 5 or not config.get("predict_angle") or config.get("cross_attention_flow_match"):
            raise ValueError("Use the Slowfast five-waypoint/angle checkpoint, not generic Qwen or OmniNav Flow")
        from model_environment import configure_model_environment
        configure_model_environment(PROJECT)
        # Imports stay here so input/coordinate checks can run without GPU libraries.
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        expected_source = (PROJECT / "third_party/OmniNav/train_code/transformers-main/src/transformers").resolve()
        if not Path(transformers.__file__).resolve().is_relative_to(expected_source):
            raise RuntimeError("Install OmniNav's pinned Transformers fork in a separate environment first")
        parameters = inspect.signature(Qwen2_5_VLForConditionalGeneration.forward).parameters
        if not {"input_waypoints", "action_former", "train_branch"} <= set(parameters):
            raise RuntimeError("The loaded Transformers class has no OmniNav waypoint-head interface")
        if not device.startswith("cuda:") or not torch.cuda.is_available():
            raise RuntimeError("Run dedicated inference only on an explicitly allocated CUDA device")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():raise RuntimeError("Selected GPU does not support BF16 inference")
            free,_=torch.cuda.mem_get_info()
        weight_bytes=sum(path.stat().st_size for path in self.checkpoint.glob("*.safetensors"))
        if free<weight_bytes+3*1024**3:
            raise RuntimeError("Insufficient free GPU memory for the checkpoint plus a 3 GiB inference reserve")
        self.model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.checkpoint), torch_dtype=torch.bfloat16, device_map="cpu" if load_on_cpu else device,
            attn_implementation=attention, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, output_loading_info=True)
        if load_on_cpu:self.model.to(device)
        head_prefixes = ("wp_predictor", "wp_predictor_angle", "input_wp_encoder", "query_action",
                         "query_multihead", "arrive_predictor")
        if any(key.startswith(head_prefixes) for key in info.get("missing_keys", [])):
            raise RuntimeError("Trained navigation-head weights are missing; refuse randomly initialized heads")
        self.model.eval()
        self.expected_arrival_count = 5 if config.get("use_arrive_list") else 1
        self.processor = AutoProcessor.from_pretrained(str(self.checkpoint), min_pixels=56*56,
                                                       max_pixels=4480*4480, local_files_only=True,
                                                       trust_remote_code=False,use_fast=False)
        self.torch = torch
        self.functions = load_functions(UPSTREAM, IMAGE_FUNCTIONS,
                                        {"np": np, "Image": Image, "math": math, "random": random})

    def predict(self, request):
        started = time.monotonic()
        images = request["images"]
        messages = [{"role": "user", "content": [{"type": "image", "image": images},
                                                    {"type": "text", "text": request["content_text"]}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
        text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        if text.count("<|image_pad|>") != 7:
            raise RuntimeError("OmniNav PointGoal must contain exactly seven image inputs")
        image_inputs, _ = self.functions["process_vision"](images)
        image_inputs = self.functions["preprocess"](image_inputs)
        inputs = self.processor(text=text, images=image_inputs, padding=True, padding_side="left", return_tensors="pt")
        inputs["input_waypoints"] = self.torch.tensor(request["input_positions_scaled"], dtype=self.torch.float32)[None]
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.forward(**inputs, action_former=True, gt_waypoints=0,
                                        train=False, train_branch=["continue"])
        if not isinstance(output, (tuple, list)) or len(output) != 4:
            raise RuntimeError("OmniNav head did not return waypoints, arrival, sin, cos")
        wp, arrival, sin_angle, cos_angle = output
        if tuple(wp.shape) != (1, 5, 2) or tuple(sin_angle.shape) != (1, 5) or tuple(cos_angle.shape) != (1, 5):
            raise RuntimeError("Unexpected dedicated head tensor shapes")
        if not all(self.torch.isfinite(tensor).all().item() for tensor in output):
            raise RuntimeError("OmniNav head returned non-finite values")
        if tuple(arrival.shape) != (1, getattr(self, "expected_arrival_count", 1)):
            raise RuntimeError("Unexpected arrival-head tensor shape")
        if self.torch.any(sin_angle.square() + cos_angle.square() < 1e-10).item():
            raise RuntimeError("OmniNav angle head returned an undefined direction")
        points = wp[0].float().cpu().tolist()
        angles = self.torch.atan2(sin_angle, cos_angle)[0].float().cpu().tolist()
        goal = native_head_first_step(points, angles, request["anchor_pose_xyyaw"])
        return {"source": "OmniNav_Slowfast_dedicated_head", "capture_time_s": request["capture_time_s"],
                "anchor_pose_xyyaw": request["anchor_pose_xyyaw"], "waypoints_normalized": points,
                "turn_angles_rad": angles, "arrival_raw": arrival.float().cpu().tolist(),
                "arrival_probabilities": arrival.sigmoid().float().cpu().tolist(),
                "arrival_used_as_task_success": False,
                "first_step_goal_xy_m": goal, "latency_s": time.monotonic() - started,
                "task_success_verified": False, "checkpoint": str(self.checkpoint)}
