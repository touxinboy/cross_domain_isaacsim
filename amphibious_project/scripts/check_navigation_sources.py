"""Reproduce narrow upstream interface issues, without model/GPU execution."""
import argparse
import ast
import json
from pathlib import Path
import re

from navigation_sources import load_functions, reviewed_tree

UPSTREAM = "third_party/OmniNav/infer_ovon_slowfast/qwen_utils.py"


def report():
    functions = load_functions(UPSTREAM, ["parse_response"], {"re": re, "json": json})
    text = "coordinate [1, 2]; not found"
    coordinate, found = functions["parse_response"](text)
    tree, _ = reviewed_tree(UPSTREAM)
    fast = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_result_fast")
    pose_writes = [node.lineno for node in ast.walk(fast)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set_state"]
    msg, _ = reviewed_tree("third_party/MSGNav/src/explore_utils.py")
    reads, writes = [], []
    for node in ast.walk(msg):
        if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                and node.slice.value == "use_prefiltering"):
            (reads if isinstance(node.ctx, ast.Load) else writes).append(node.lineno)
    return {"scope": "offline_source_and_parser_checks_not_navigation_evaluation", "model_inference_run": False,
            "issues": [
                {"id": "OMNI_PARSE_NEGATED_FOUND", "method": "executed original parse_response",
                 "test_input": text, "returned_coordinate": coordinate, "returned_found": found,
                 "expected_found": False, "reproduced": found is True,
                 "impact": "The substring 'found' also matches 'not found'. Not used for success decisions by this adapter."},
                {"id": "OMNI_DEMO_POSE_WRITES", "method": "static source check", "lines": sorted(pose_writes),
                 "impact": "Original demo includes direct set_state motion and unsticking; excludes propulsion/contact dynamics. Adapter excludes these branches."},
                {"id": "MSG_PREFILTER_FLAG", "method": "static source check in explore_utils.py",
                 "use_prefiltering_store_lines": writes, "use_prefiltering_read_lines": reads,
                 "impact": "The inspected decision module stores this flag but does not read it; KSS still calls prefiltering. Not a full-repository claim."}
            ],
            "integration_limits": ["Isaac perception and frontier construction are pending",
                                   "Unknown image goals need detector/depth localization before PointGoal",
                                   "Qwen service and dedicated OmniNav checkpoint have not run",
                                   "No water/shore traversability or learned mode switching is provided by these upstream models"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = report()
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
