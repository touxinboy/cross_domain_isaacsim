"""One real seven-image waypoint inference on an explicitly selected server GPU.

Synthetic input images exercise the full trained checkpoint's API only. The
result is explicitly marked as a fixture and cannot drive the live session.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/"scripts"))
from navigation_io import write_json


def smoke_with_predictor(predictor):
    from omninav_pointgoal import UPSTREAM
    from navigation_sources import load_string_assignment
    import numpy as np
    predictor.torch.cuda.reset_peak_memory_stats(predictor.model.device)
    width,height=json.loads((PROJECT/'configs/navigation_camera.json').read_text())['resolution_wh']
    images=[np.full((height,width,3),[20+20*i,100,180-10*i],dtype=np.uint8) for i in range(7)]
    request={"capture_time_s":5.,"anchor_pose_xyyaw":[0.,0.,0.],"images":images,
             "input_positions_scaled":[[0.,0.]]*5+[[0.,2.]],
             "content_text":load_string_assignment(UPSTREAM,"get_result_fast","content_text")}
    result=predictor.predict(request);result["test_fixture_only"]=True
    return {"passed":True,"test_fixture_only":True,"result":result,
            "validation_scope":"synthetic_images_real_trained_checkpoint_forward_not_navigation_performance",
            "peak_allocated_bytes":predictor.torch.cuda.max_memory_allocated(predictor.model.device)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=PROJECT/"checkpoints/OmniNav_Slowfast")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--attention",choices=["eager","sdpa","flash_attention_2"],default="eager")
    parser.add_argument("--output",type=Path,default=PROJECT/"results/deployment/model_gpu_smoke.json")
    args=parser.parse_args()
    report={"validation_scope":"synthetic_images_real_trained_checkpoint_GPU_forward_not_navigation_performance",
            "robot_motion_started":False,"test_fixture_only":True,"passed":False}
    try:
        from omninav_pointgoal import OmniNavPointGoal
        predictor=OmniNavPointGoal(args.checkpoint,args.device,args.attention)
        report.update(smoke_with_predictor(predictor))
    except Exception:report["error"]=traceback.format_exc()
    write_json(args.output,report)
    print("GPU checkpoint smoke passed:",report["passed"],"; report:",args.output)
    return 0 if report["passed"] else 3


if __name__=="__main__":raise SystemExit(main())
