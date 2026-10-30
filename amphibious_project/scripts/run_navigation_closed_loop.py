"""Prepare/send an own GUI experiment request; never launches Isaac or a GPU job."""
import argparse
from datetime import datetime
from pathlib import Path
import time
import uuid

from navigation_io import own_path, read_json, validate_goal, write_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", default="configs/navigation_goal.example.json")
    parser.add_argument("--mode", choices=["shadow", "execute"], default="shadow")
    parser.add_argument("--send", action="store_true", help="Ask the existing own GUI to start; otherwise prepare only")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    current = PROJECT/".runtime/navigation/current_request.json"
    if args.stop or args.status:
        request = read_json(current)
        path = own_path(PROJECT, request["output_relative"])/"summary.json"
        if args.stop: write_json(own_path(PROJECT,request["stop_relative"]),{"reason":"user_stop"})
        elif path.exists(): print(path.read_text())
        else:
            progress = path.with_name("progress.json")
            print(progress.read_text() if progress.exists() else "Prepared or waiting; no completed summary yet.")
        return
    goal = read_json(own_path(PROJECT,args.goal)); validate_goal(goal)
    run_id = "run_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
    request = {"schema":"L17.navigation_run.v1","run_id":run_id,"mode":args.mode,"goal":goal,
               "start_deadline_wall_unix_s":time.time()+300,
               "output_relative":"results/navigation/"+run_id,"stop_relative":".runtime/navigation/"+run_id+".stop.json"}
    write_json(current,request)
    print("Prepared:",current)
    if not args.send:
        print("No simulator/model process was started. Use start_navigation_closed_loop.py in the own GUI.")
        return
    directory = PROJECT/".runtime/robot_ui"
    if not directory.is_dir(): raise RuntimeError("Open the own L17 GUI before sending a request")
    name = "navigation_"+uuid.uuid4().hex
    write_json(directory/(name+".request.json"),{"action":"navigation_start","request":request})
    deadline = time.monotonic()+15
    result_path = directory/(name+".result.json")
    while time.monotonic()<deadline:
        if result_path.exists():
            result=read_json(result_path)
            if not result.get("ok"): raise RuntimeError(result.get("error","GUI rejected the request"))
            print("Own GUI accepted:",request["run_id"]);return
        time.sleep(.1)
    raise RuntimeError("Own GUI did not reply; reload the updated own panel once in the Script Editor")


if __name__=="__main__": main()
