"""Bind an MSGNav image selection, 2-D grounding and the exact RGB-D frame.

Produces surface evidence only. A separate approach planner must create a
traversable robot arrival pose before the waypoint worker is allowed to run.
"""
import argparse
from pathlib import Path

from depth_localization import DepthFrame, localize_surface
from navigation_io import file_sha, own_path, read_json, write_json


def selected_frame(observation, decision, image_root):
    if observation.get("source") not in ["perception", "synthetic_fixture"]:
        raise ValueError("Use observed images rather than hidden scene objects")
    if decision.get("source") != "MSGNav_original_decision_module" or decision.get("choice_type") != "image":
        raise ValueError("Depth resolution currently supports an MSGNav image selection")
    if abs(float(observation["capture_time_s"]) - float(decision["capture_time_s"])) > 1e-6:
        raise ValueError("Decision and observation refer to different times")
    evidence = decision["evidence"]
    if isinstance(evidence, str):
        snapshot = observation["snapshots"][evidence]
        image, frame_path = snapshot["image"], snapshot.get("rgbd_frame")
    elif isinstance(evidence, dict) and type(evidence.get("egocentric_index")) is int:
        index = evidence["egocentric_index"]
        if not 0 <= index < len(observation.get("egocentric_images", [])): raise ValueError("Invalid egocentric image index")
        image = observation["egocentric_images"][index]
        frames = observation.get("egocentric_rgbd_frames", [])
        frame_path = frames[index] if index < len(frames) else None
    else: raise ValueError("Unsupported MSGNav image evidence")
    if not frame_path: raise ValueError("Selected image has no matching RGB-D frame; do not use a different depth capture")
    frame_path = own_path(image_root, frame_path)
    frame = DepthFrame.load(frame_path)
    if frame.metadata.get("camera_role") != "robot_mounted": raise ValueError("Use robot-mounted observations for navigation")
    rgb = own_path(image_root, image)
    if file_sha(rgb) != frame.metadata["rgb_sha256"]: raise ValueError("Selected image differs from the depth frame's RGB")
    return rgb, frame


def resolve(observation, decision, grounding, image_root):
    rgb, frame = selected_frame(observation, decision, image_root)
    surface = localize_surface(frame, grounding)
    return {"schema": "L17.selected_surface.v1", "selection": decision["raw_response"],
            "selected_rgb": str(rgb), "grounding": grounding, "surface": surface,
            "goal_resolved": False, "goal_xy_m": None,
            "next_required_step": "observed_free_space_and_standoff_arrival_planning",
            "robot_motion_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, help="Replay a saved grounding on its exact image")
    parser.add_argument("--use-qwen", action="store_true", help="Explicitly make a new image API request")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.grounding) == args.use_qwen: parser.error("Choose a saved grounding or --use-qwen")
    observation, decision = read_json(args.observation), read_json(args.decision)
    if args.output.resolve() in [p.resolve() for p in [args.observation, args.decision, args.grounding] if p]:
        raise ValueError("Preserve all input artifacts")
    if args.use_qwen:
        from ground_target_api import ground
        from qwen_api_client import QwenClient
        rgb, _ = selected_frame(observation, decision, args.image_root)
        grounding = ground(QwenClient(timeout_s=120), rgb, decision.get("category") or observation["instruction"])
    else: grounding = read_json(args.grounding)
    result = resolve(observation, decision, grounding, args.image_root)
    write_json(args.output, result)
    print("Surface accepted:", result["surface"]["accepted"], "; arrival goal still requires approach planning")
    return 0 if result["surface"]["accepted"] else 3


if __name__ == "__main__": raise SystemExit(main())
