"""Validate a recorded perception snapshot or run MSGNav/Qwen on that snapshot.

No robot movement is started by this command. Output goals are interface data,
not evidence of successful navigation. Model errors are never replaced by a
scripted crossing route.
"""
import argparse
import json
from pathlib import Path
import sys

from msgnav_bridge import MSGNavDecision, build_step
from omninav_pointgoal import OmniNavPointGoal, prepare_request
from qwen_api_client import QwenAPIError, QwenClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true", help="No HTTP requests")
    parser.add_argument("--output", type=Path, help="Output JSON; defaults beside the observation")
    parser.add_argument("--omninav-checkpoint", type=Path, help="Optional local Slowfast inference after MSGNav; uses allocated GPU")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    path = args.observation.resolve()
    observation = json.loads(path.read_text(encoding="utf-8-sig"))
    if args.validate_only:
        step, capture, _ = build_step(observation, path.parent)
        result = {"status": "input_validated_no_inference", "source": observation["source"],
                  "capture_time_s": capture, "objects": len(step["objects"]), "edges": len(step["edges"]),
                  "snapshots": len(step["all_imgs"]), "frontiers": len(step["frontier_imgs"]),
                  "task_success_verified": False}
        if observation.get("pointgoal"):
            target = observation.get("validation_pointgoal_xy_m", observation["anchor_pose_xyyaw"][:2])
            prepare_request(observation, target, path.parent)
            result["pointgoal_input_validated"] = True
    else:
        client = QwenClient()
        result = MSGNavDecision(client).decide(observation, path.parent)
        if args.omninav_checkpoint:
            if not result["goal_resolved"]:
                raise RuntimeError("MSGNav selection needs observed geometric localization before OmniNav can run")
            request = prepare_request(observation, result["goal_xy_m"], path.parent)
            predictor = OmniNavPointGoal(args.omninav_checkpoint, device=args.device)
            result["pointgoal_result"] = predictor.predict(request)
    output = args.output or path.with_name(path.stem + ".decision.json")
    if output.resolve() == path:
        raise ValueError("Output must not overwrite the input observation")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "status": result.get("status", "decision_recorded"),
                      "robot_motion_started": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError, QwenAPIError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(3)
