"""Diagnose the isolated waypoint environment; model weights are never loaded."""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/"scripts"))
from navigation_io import write_json
from prepare_checkpoint import inspect_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=PROJECT/"checkpoints/OmniNav_Slowfast")
    parser.add_argument("--check-cuda",action="store_true",help="Read GPU capability; do not allocate model weights")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    from model_environment import configure_model_environment
    configure_model_environment(PROJECT)
    result={"python":sys.version,"executable":sys.executable,"model_loaded":False,"errors":[],"versions":{}}
    for name in ["torch","torchvision","transformers","diffusers","numpy","accelerate","safetensors","tokenizers"]:
        try:result["versions"][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result["errors"].append("Missing package: "+name)
    try:
        import transformers
        from transformers import Qwen2_5_VLForConditionalGeneration
        expected=(PROJECT/"third_party/OmniNav/train_code/transformers-main/src/transformers").resolve()
        result["transformers_path"]=transformers.__file__
        if not Path(transformers.__file__).resolve().is_relative_to(expected):raise RuntimeError("Not the pinned OmniNav Transformers fork")
        if not {"input_waypoints","action_former","train_branch"}<=set(inspect.signature(Qwen2_5_VLForConditionalGeneration.forward).parameters):
            raise RuntimeError("Dedicated waypoint-head forward parameters are missing")
    except Exception as error:result["errors"].append(str(error))
    try:result["checkpoint"]=inspect_checkpoint(args.checkpoint.resolve())
    except Exception as error:result["errors"].append(str(error))
    result["software_and_checkpoint_ready"]=not result["errors"]
    result["cuda_check_requested"]=args.check_cuda
    if args.check_cuda:
        try:
            import torch
            result["cuda_available"]=torch.cuda.is_available()
            if not result["cuda_available"]:raise RuntimeError("CUDA is unavailable")
            result["visible_cuda_devices"]=torch.cuda.device_count()
            result["device0"]=torch.cuda.get_device_name(0)
            result["bf16_supported"]=torch.cuda.is_bf16_supported()
            if not result["bf16_supported"]:raise RuntimeError("Selected GPU does not support the required BF16 dtype")
        except Exception as error:result["errors"].append(str(error))
    result["ready_for_model_load"]=(not result["errors"]) if args.check_cuda else None
    if args.output:write_json(args.output,result)
    print(json.dumps(result,indent=2))
    return 0 if not result["errors"] else 3


if __name__=="__main__":raise SystemExit(main())
