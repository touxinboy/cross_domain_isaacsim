"""Foreground waypoint worker launcher; never requests a cluster allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

PROJECT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu",nargs="?",type=int,help="Explicit available GPU index on the interactive Isaac machine")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    env=os.environ.copy()
    if env.get("SLURM_JOB_ID"):
        if args.gpu is not None:parser.error("Inside Slurm, omit the GPU argument to preserve the allocated CUDA mask")
    else:
        if args.gpu is None or args.gpu<0:parser.error("Choose an available GPU index explicitly")
        env["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
        if PROJECT.is_relative_to(Path("/home/wangjiong/L17")):
            raise RuntimeError("The EPIC cluster worker must run inside an already allocated Slurm GPU job")
    command=[str(PROJECT/".venvs/omninav/bin/python"),"-B",str(PROJECT/"scripts/omninav_worker.py"),
             "--checkpoint","checkpoints/OmniNav_Slowfast","--device","cuda:0","--attention","eager",
             "--idle-exit-s","600","--recover-stale-lock","--startup-smoke"]
    if args.dry_run:
        print(json.dumps({"command":command,"CUDA_VISIBLE_DEVICES":env.get("CUDA_VISIBLE_DEVICES"),"gpu_job_requested":False},indent=2));return 0
    if os.name!="posix":raise RuntimeError("Use this launcher on the target Linux server")
    roots=[Path("/home/ubuntu/sdc/L17"),Path("/home/wangjiong/L17")]
    if not any(PROJECT.is_relative_to(root.resolve()) for root in roots if root.exists()):
        raise RuntimeError("Start the worker only from the own L17 project")
    return subprocess.call(command,cwd=PROJECT,env=env)


if __name__=="__main__":raise SystemExit(main())
