#!/usr/bin/env python3
"""Prepare/check/start/stop only L17's own full river-crossing experiment.

System Python launcher. It does not install packages, change shared Isaac Sim,
terminate processes, or connect to any server. --start is an explicit local action
on the Linux host after upload. Default closes ONLY the new app after export.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid


PROJECT = Path(__file__).resolve().parents[1]
L17 = Path("/home/ubuntu/sdc/L17")
DEPENDENCIES = ["scripts/launch.py", "scripts/robot_runtime.py", "scripts/amphibian_dynamics.py",
                "scripts/robot_assets.py", "scripts/river_geometry.py", "scripts/crossing_control.py",
                "scripts/crossing_experiment.py", "scripts/crossing_report.py", "scripts/start_crossing_experiment.py",
                "configs/robot_v1.json", "configs/river_forest.json", "configs/crossing_experiment.json",
                "configs/task.json", "scenes/river_robot_v1.usda", "scenes/river_forest_v1.usda",
                "robots/tracked_twin_thruster_v1.usda"]


def inside_project(relative):
    path = (PROJECT / relative).resolve()
    if not path.is_relative_to(PROJECT):
        raise RuntimeError("Path escapes the own project")
    return path


def server_guard():
    if sys.platform != "linux":
        raise RuntimeError("Run --start/--stop/--status on the Linux server after upload. --check is available locally.")
    root = L17.resolve(strict=True)
    if root != L17 or not PROJECT.is_relative_to(root):
        raise RuntimeError("The project must stay inside /home/ubuntu/sdc/L17")


def file_check():
    checks = {path: inside_project(path).is_file() for path in DEPENDENCIES}
    cfg = json.loads(inside_project("configs/crossing_experiment.json").read_text())
    return {"files_present": all(checks.values()), "files": checks, "mission": cfg["mission"],
            "runtime_not_executed": True, "server_connection_performed": False,
            "physics_hz": cfg["physics_hz"], "control_hz": cfg["control_hz"]}


def active_request():
    target = inside_project(".runtime/crossing_active.json")
    if not target.is_file():
        raise RuntimeError("No recorded crossing run. Use --start first.")
    data = json.loads(target.read_text())
    if not re.fullmatch(r"run_[0-9]{8}_[0-9]{6}_[a-f0-9]{8}", data["run_id"]):
        raise RuntimeError("Invalid own run id")
    if data["output_relative"] != "results/crossing/" + data["run_id"]:
        raise RuntimeError("Invalid result directory")
    if data["stop_relative"] != ".runtime/crossing/" + data["run_id"] + ".stop":
        raise RuntimeError("Invalid stop-file destination")
    return data


def mark_stop(request):
    path = inside_project(request["stop_relative"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("User requested a stop for " + request["run_id"] + "\n", encoding="utf-8")
    print("Stop requested for own run " + request["run_id"], flush=True)


def gpu_check(max_memory, max_util):
    cfg = json.loads(inside_project("configs/task.json").read_text())
    gpu = cfg.get("gui_gpu_index", 0)
    result = subprocess.run(["nvidia-smi", "-i", str(gpu),
                             "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
    name, used, total, util = [x.strip() for x in result.stdout.strip().split(",")]
    snapshot = {"gpu_index": gpu, "name": name, "memory_used_mib": float(used),
                "memory_total_mib": float(total), "gpu_utilization_percent": float(util)}
    if float(used) > max_memory or float(util) > max_util:
        raise RuntimeError("GPU is busy; no app was started and no process was stopped. " + json.dumps(snapshot))
    # Avoid racing an existing own GUI or colliding with its private cache/runtime.
    processes = subprocess.run(["ps", "-C", "kit", "-o", "pid=,args="], capture_output=True, text=True, timeout=10)
    own_pids = [line.strip().split()[0] for line in processes.stdout.splitlines()
                if "--portable-root " + str(PROJECT / ".runtime/kit") in line]
    if own_pids:
        raise RuntimeError("An own L17 Isaac window is still open (PID " + ",".join(own_pids)
                           + "). Close that own window before starting this test. No process was stopped.")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Read-only local file checks; uses no GPU")
    action.add_argument("--start", action="store_true", help="Start this test on the Linux server")
    action.add_argument("--status", action="store_true", help="Read latest own progress/result")
    action.add_argument("--stop", action="store_true", help="Request own active crossing test to stop")
    parser.add_argument("--keep-open", action="store_true", help="Keep the own final GUI paused for inspection")
    parser.add_argument("--max-gpu-memory-mib", type=float, default=2048.)
    parser.add_argument("--max-gpu-utilization", type=float, default=25.)
    args = parser.parse_args()
    if args.check:
        report = file_check()
        print(json.dumps(report, indent=2))
        return 0 if report["files_present"] else 3
    server_guard()
    if args.status or args.stop:
        request = active_request()
        output = inside_project(request["output_relative"])
        if args.stop:
            if (output / "summary.json").is_file():
                print("Own run already ended; no stop request was written.")
            else:
                mark_stop(request)
        else:
            target = output / ("summary.json" if (output / "summary.json").is_file() else "progress.json")
            print(target.read_text() if target.is_file() else "Starting. See launcher log under .runtime/logs/.")
        return 0
    if not file_check()["files_present"]:
        raise RuntimeError("Project files are missing; run --check")
    import fcntl
    runtime = inside_project(".runtime")
    runtime.mkdir(exist_ok=True)
    with (runtime / "crossing.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An own crossing launcher is already active. Use --status or --stop.")
        snapshot = gpu_check(args.max_gpu_memory_mib, args.max_gpu_utilization)
        run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        output_relative = "results/crossing/" + run_id
        output = inside_project(output_relative)
        output.mkdir(parents=True, exist_ok=False)
        request = {"schema_version": 1, "kind": "reference_physical_crossing", "run_id": run_id,
                   "output_relative": output_relative, "stop_relative": ".runtime/crossing/" + run_id + ".stop",
                   "close_when_done": not args.keep_open, "launcher_pid": os.getpid(), "gpu_before_start": snapshot,
                   "created": datetime.now().astimezone().isoformat()}
        request_text = json.dumps(request, indent=2) + "\n"
        for target in [runtime / "crossing_active.json", runtime / "crossing_pending.json", output / "request.json"]:
            temporary = target.with_suffix(".tmp")
            temporary.write_text(request_text, encoding="utf-8")
            temporary.replace(target)
        inside_project(request["stop_relative"]).parent.mkdir(exist_ok=True)
        def interrupt(signum, frame):
            mark_stop(request)
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        print("Own run: " + run_id + "\nResults: " + str(output), flush=True)
        command = [sys.executable, "-B", str(PROJECT / "scripts/launch.py"), "gui",
                   "--physics-gpu-index", str(snapshot["gpu_index"]),
                   "--exec-script", "scripts/start_crossing_experiment.py"]
        completed = subprocess.run(command, cwd=str(PROJECT))
        summary_file = output / "summary.json"
        if not summary_file.is_file():
            print("No physics result. Check .runtime/crossing_startup_error.txt and .runtime/logs/.", file=sys.stderr)
            return completed.returncode or 3
        summary = json.loads(summary_file.read_text())
        print(json.dumps({"result": summary["result"], "crossing_passed": summary["crossing_passed"],
                          "summary": str(summary_file)}, indent=2))
        return 0 if summary["crossing_passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print("CROSSING_NOT_STARTED: " + str(error), file=sys.stderr)
        raise SystemExit(3)
