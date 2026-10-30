"""Persistent local-file waypoint service in an isolated model environment.

Run separately from Isaac. No server, GPU reservation, network listener or
checkpoint download is started automatically. Raw inputs and outputs remain
available for investigating model failures.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import time
import traceback
import uuid
from contextlib import nullcontext

from navigation_io import file_sha, own_path, read_json, validate_goal, write_json, waypoint_code_identity
from omninav_pointgoal import OmniNavPointGoal, prepare_request


def process_request(project, request_path, predictor):
    project = Path(project).resolve()
    request_path = own_path(project, request_path)
    request_id=request_path.name.removesuffix(".request.json")
    reply = {"schema": "L17.waypoint_result.v1", "request_id":request_id,
             "controller_epoch":None,"observation_sha256":None,"ok": False,
             "worker_pid": os.getpid(), "model_inference_run": False,"model_inference_attempted":False}
    started = time.monotonic()
    try:
        request=read_json(request_path)
        if request.get("schema") != "L17.waypoint_request.v1":raise ValueError("Invalid waypoint request schema")
        if request.get("request_id") != request_id:raise ValueError("Request filename and identity differ")
        for key in ["controller_epoch","observation_sha256"]:reply[key]=request[key]
        if time.time()>float(request["deadline_wall_unix_s"]): raise ValueError("Request expired before model inference")
        if request_path.with_name(request["request_id"]+".cancelled.json").exists():raise ValueError("Request was cancelled")
        observation_path = own_path(project, request["observation_file"])
        if file_sha(observation_path) != request["observation_sha256"]: raise ValueError("Observation file was modified")
        observation = read_json(observation_path)
        if observation.get("controller_epoch") != request["controller_epoch"]: raise ValueError("Runtime epoch mismatch")
        goal_xy = validate_goal(request["goal"])
        prepared = prepare_request(observation, goal_xy, own_path(project, request["image_root"]))
        reply["model_inference_attempted"]=True
        reply["result"] = predictor.predict(prepared)
        reply["model_inference_run"] = not bool(reply["result"].get("test_fixture_only"))
        reply["ok"] = True
    except Exception:
        reply["error"] = traceback.format_exc()
    reply["worker_wall_s"] = time.monotonic() - started
    output = request_path.with_name(request_id + ".result.json")
    write_json(output, reply)
    return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--queue", default=".runtime/navigation/worker")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--idle-exit-s", type=float, default=600)
    parser.add_argument("--recover-stale-lock",action="store_true",help="Linux only: reclaim the own lock only if its recorded PID no longer exists")
    parser.add_argument("--startup-smoke",action="store_true",help="Require a real seven-image forward pass before publishing ready.json")
    parser.add_argument("--offload-between-requests",action="store_true",help="Serialize own model GPU residency and retain idle weights in CPU RAM")
    args = parser.parse_args()
    if not math.isfinite(args.idle_exit_s) or args.idle_exit_s<=0:parser.error("idle-exit-s must be finite and positive")
    project = args.project.resolve()
    queue = own_path(project, args.queue)
    queue.mkdir(parents=True, exist_ok=True)
    # An exclusive lock is released on normal/error exit. After a hard crash,
    # inspect the PID before removing this own-project lock; never kill by name.
    lock = queue / "worker.lock"
    if (queue/"STOP").exists():raise RuntimeError("Own queue has a STOP marker; clear that marker before intentionally restarting the worker")
    if lock.exists() and args.recover_stale_lock:
        if os.name!="posix":raise RuntimeError("Automatic stale-lock recovery is Linux-only")
        previous=read_json(lock)
        previous_pid=int(previous["pid"]) if isinstance(previous,dict) else int(previous)
        if previous_pid<=0:raise ValueError("Invalid old worker PID")
        try:os.kill(previous_pid,0)
        except ProcessLookupError:lock.unlink()
        else:raise RuntimeError("Recorded worker PID is still alive; no process was modified")
    worker_id = uuid.uuid4().hex
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor,json.dumps({"pid":os.getpid(),"worker_id":worker_id}).encode());os.close(descriptor)
    try:
        (queue/"ready.json").unlink(missing_ok=True)
        (queue/"heartbeat.json").unlink(missing_ok=True)
        checkpoint=(project/args.checkpoint).resolve()
        code_identity=waypoint_code_identity(project)
        smoke_passed=False
        from gpu_lease import own_gpu_lease,offload_model,OffloadedPredictor,residency_state
        with own_gpu_lease(project) if args.offload_between_requests else nullcontext():
            predictor = OmniNavPointGoal(checkpoint, args.device, args.attention,load_on_cpu=args.offload_between_requests)
            try:
                if args.startup_smoke:
                    import importlib.util
                    source=project/"deploy/omninav/smoke_model.py"
                    spec=importlib.util.spec_from_file_location("l17_waypoint_gpu_smoke",source)
                    smoke_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke_module)
                    smoke=smoke_module.smoke_with_predictor(predictor)
                    write_json(queue/"startup_smoke.json",smoke)
                    smoke_passed=bool(smoke["passed"])
                    if not smoke_passed:raise RuntimeError("Startup model inference smoke failed")
            finally:
                if args.offload_between_requests:offload_model(predictor)
        idle_residency=residency_state(predictor) if args.offload_between_requests else None
        if idle_residency:
            write_json(queue/'idle_residency.json',idle_residency)
            print('L17_WAYPOINT_IDLE_RESIDENCY',json.dumps(idle_residency),flush=True)
        if args.offload_between_requests:predictor=OffloadedPredictor(project,predictor,args.device)
        ready = {"worker_id": worker_id,"project":str(project),"code_sha256":code_identity,
                 "checkpoint_config_sha256":file_sha(checkpoint/"config.json"),
                 "pid": os.getpid(), "checkpoint": str(checkpoint),
                 "device": args.device, "attention": args.attention, "model_loaded": True,"startup_smoke_passed":smoke_passed,
                 "weights_offloaded_to_CPU_between_requests":args.offload_between_requests,
                 "idle_residency":idle_residency,
                 "started_utc": datetime.now(timezone.utc).isoformat()}
        write_json(queue / "ready.json", ready)
        write_json(queue / "heartbeat.json", {**ready, "wall_unix_s": time.time()})
        print("L17_WAYPOINT_WORKER_READY",json.dumps({"worker_id":worker_id,"device":args.device,"startup_smoke_passed":smoke_passed}),flush=True)
        last_work = last_heartbeat = time.monotonic()
        while time.monotonic() - last_work < args.idle_exit_s:
            if (queue / "STOP").exists(): break
            for path in sorted(queue.glob("*.request.json")):
                result_path = path.with_name(path.name.removesuffix(".request.json") + ".result.json")
                if result_path.exists(): continue
                if path.with_name(path.name.removesuffix(".request.json")+".cancelled.json").exists():continue
                try: reply=process_request(project, path, predictor)
                except Exception:
                    reply={"ok":False}
                    write_json(queue / (path.stem + ".protocol_error.json"), {"error": traceback.format_exc()})
                last_work = time.monotonic()
                if args.once: return 0 if reply["ok"] else 3
            if time.monotonic() - last_heartbeat > 1:
                write_json(queue / "heartbeat.json", {**ready, "wall_unix_s": time.time()})
                last_heartbeat = time.monotonic()
            time.sleep(.05)
    except BaseException:
        write_json(queue/"startup_or_runtime_error.json",{"worker_id":worker_id,"error":traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)
        write_json(queue / "stopped.json", {"worker_id": worker_id, "pid": os.getpid(), "wall_unix_s": time.time()})


if __name__ == "__main__":raise SystemExit(main())
