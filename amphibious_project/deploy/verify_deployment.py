"""Portable preflight for an unpacked project and separately transferred weights.

Uses standard Python. With OpenUSD available, also composes the actual scene.
No network access, model inference, GPU reservation or robot motion is started.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"scripts"))
from navigation_io import own_path,read_json,write_json


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def verify_manifest(project,required=False):
    path=project/"PACKAGED_FILES.json"
    if not path.exists():
        if required:raise ValueError("Unpack the prepared archive first; its file manifest is missing")
        return {"status":"not_applicable_source_checkout"}
    entries=read_json(path)["files"]
    for entry in entries:
        file=own_path(project,entry["path"])
        if not file.is_file() or file.stat().st_size!=entry["bytes"] or sha(file)!=entry["sha256"]:
            raise ValueError("Packaged file is absent or modified: "+entry["path"])
    return {"status":"passed","checked_files":len(entries)}


def verify_sources(project):
    parsed=0
    for name in ["scripts","deploy","tests"]:
        for path in (project/name).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path),feature_version=(3,10));parsed+=1
    manifest=read_json(project/"configs/navigation_sources.json")
    for name,expected in manifest["files"].items():
        if sha(own_path(project,name))!=expected:raise ValueError("Pinned upstream source changed: "+name)
    for folder in ["scripts","deploy"]:
        for path in (project/folder).rglob("*.sh"):
            if b"\r\n" in path.read_bytes():raise ValueError("Linux shell script contains Windows CRLF endings: "+str(path))
    return {"status":"passed","own_python_files_python310_syntax":parsed,"pinned_upstream_files":len(manifest["files"])}


def verify_scene(project):
    task=read_json(project/"configs/task.json")
    scene=own_path(project,task["scene_file"])
    if not scene.is_file():raise ValueError("Task scene is missing")
    references=0
    for folder in ["scenes","robots","assets"]:
        for layer in (project/folder).rglob("*.usda"):
            for value in re.findall(r"@([^@]+)@",layer.read_text()):
                if "://" in value:raise ValueError("Scene needs an external asset: "+value)
                target=own_path(project,layer.parent/value)
                if not target.is_file():raise ValueError("Missing USD reference: "+value)
                lexical=Path(os.path.normpath(str(layer.parent/value)))
                current=project
                for part in lexical.relative_to(project).parts:
                    if part not in {p.name for p in current.iterdir()}:raise ValueError("USD path letter-case mismatch: "+value)
                    current=current/part
                references+=1
    report={"status":"passed_asset_paths","scene":task["scene_file"],"references_checked":references}
    try:from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils
    except ImportError:report["OpenUSD_composition"]="not_run_use_own_Isaac_GUI_sensor_check";return report
    stage=Usd.Stage.Open(str(scene))
    layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(scene))
    if not stage or unresolved:raise ValueError("USD composition failed or has unresolved assets: "+str(unresolved))
    if UsdGeom.GetStageMetersPerUnit(stage)!=1 or str(UsdGeom.GetStageUpAxis(stage))!="Z":raise ValueError("Stage must use metres and Z-up")
    robots=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(robots)!=1 or str(robots[0].GetPath())!="/World/Robot":raise ValueError("Expected one own robot rigid body")
    expected_hz=read_json(project/"configs/robot_v1.json")["physics_hz"]
    physics_hz=stage.GetPrimAtPath("/World/PhysicsScene").GetAttribute("physxScene:timeStepsPerSecond").Get()
    if physics_hz!=expected_hz:raise ValueError("Composed physics frequency differs from robot configuration")
    report.update(status="passed_composition",OpenUSD_composition="passed",layers=len(layers),unresolved_assets=[],
                  prim_count=sum(1 for _ in stage.Traverse()),meters_per_unit=1,up_axis="Z",physics_hz=physics_hz)
    return report


def verify_weights(checkpoint,lock):
    for entry in lock["files"]:
        path=own_path(checkpoint,entry["path"])
        if not path.is_file():raise ValueError("Missing model file; transfer the separate checkpoint directory: "+entry["path"])
        if path.stat().st_size!=entry["bytes"] or sha(path)!=entry["sha256"]:raise ValueError("Model size/SHA-256 mismatch: "+entry["path"])
    return {"status":"passed","files":len(lock["files"]),"revision":lock["revision"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=PROJECT/"checkpoints/OmniNav_Slowfast")
    parser.add_argument("--require-manifest",action="store_true")
    parser.add_argument("--skip-weights",action="store_true",help="Code-only check; not sufficient for model startup")
    parser.add_argument("--output",type=Path,default=PROJECT/"results/deployment/preflight.json")
    args=parser.parse_args()
    result={"project":str(PROJECT),"gpu_used":False,"model_inference_run":False,"errors":[]}
    checks={"package":lambda:verify_manifest(PROJECT,args.require_manifest),"sources":lambda:verify_sources(PROJECT),
            "scene":lambda:verify_scene(PROJECT)}
    if not args.skip_weights:checks["checkpoint"]=lambda:verify_weights(args.checkpoint.resolve(),read_json(PROJECT/"configs/omninav_checkpoint.lock.json"))
    for name,check in checks.items():
        try:result[name]=check()
        except Exception as error:result["errors"].append({"check":name,"error":str(error)})
    result["static_preflight_passed"]=not result["errors"]
    result["server_GPU_and_PhysX_validation"]="pending_target_machine"
    write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result["static_preflight_passed"] else 3


if __name__=="__main__":raise SystemExit(main())
