"""Small, inspectable file contracts shared by Kit and a separate model worker."""
import hashlib
import json
from pathlib import Path
import uuid


def own_path(project, relative):
    root = Path(project).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError("Navigation files must remain inside this project")
    return path


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 4_000_000: raise ValueError("Navigation JSON exceeds the interface limit")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def waypoint_code_identity(project):
    root=Path(project).resolve()
    names=["scripts/omninav_pointgoal.py","scripts/navigation_contracts.py",
           "third_party/OmniNav/infer_ovon_slowfast/qwen_utils.py","configs/navigation_camera.json"]
    return {name:file_sha(root/name) for name in names}


def validate_goal(goal):
    from navigation_contracts import finite_vector
    if goal.get("kind") != "robot_arrival_pose":
        raise ValueError("A visible target surface is not a robot arrival pose")
    if goal.get("source") not in ["observed_traversability", "manual_pointgoal_interface_test"]:
        raise ValueError("Declare an observed-map goal or an explicit manual PointGoal test")
    xy = finite_vector(goal["goal_xy_m"], 2, "arrival goal")
    if goal["source"] == "observed_traversability" and not goal.get("approach_validated"):
        raise ValueError("An observed target needs a traversability-checked approach pose")
    return xy
