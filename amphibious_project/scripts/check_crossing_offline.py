"""Check geometry/controller inputs offline. Does NOT run or emulate PhysX."""

import argparse
import json
from pathlib import Path
import sys

from crossing_control import route_preflight
from river_geometry import RiverGeometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    cfg = json.loads((project / "configs/crossing_experiment.json").read_text())
    scene = json.loads((project / "configs/river_forest.json").read_text())
    robot = json.loads((project / "configs/robot_v1.json").read_text())
    result = route_preflight(cfg, scene, robot, RiverGeometry(scene))
    target = Path(args.output).resolve()
    allowed = Path("/home/ubuntu/sdc/L17").resolve(strict=True) if sys.platform == "linux" else project.parents[2]
    if not target.is_relative_to(allowed):
        raise RuntimeError("Offline check output must remain in the local workspace/L17 tree")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ["profile", "waypoints_xy_m"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
