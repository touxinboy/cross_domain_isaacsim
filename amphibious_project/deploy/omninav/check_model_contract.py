"""CPU-only verification of real source/processor/checkpoint structure.

The full architecture is built on meta tensors. The optional small forward
uses random small weights and the real tokenizer/preprocessor; it is an API
compatibility test, never a learned navigation result.
"""
import argparse
import gc
import json
from pathlib import Path
import struct
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/"scripts"))
from navigation_io import write_json


def checkpoint_schema(checkpoint):
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
    raw=json.loads((checkpoint/"config.json").read_text())
    config=Qwen2_5_VLConfig.from_dict(raw)
    config._attn_implementation="eager"
    with init_empty_weights(include_buffers=True):model=Qwen2_5_VLForConditionalGeneration(config)
    expected={k:list(v.shape) for k,v in model.state_dict().items()}
    actual={};heads={}
    prefixes=("wp_predictor","input_wp_encoder","query_action","query_multihead","arrive_predictor")
    index=json.loads((checkpoint/"model.safetensors.index.json").read_text())["weight_map"]
    for shard in sorted(set(index.values())):
        path=checkpoint/shard
        with path.open("rb") as handle:
            size=struct.unpack("<Q",handle.read(8))[0]
            if not 2<=size<=16_000_000:raise ValueError("Invalid safetensors header")
            header=json.loads(handle.read(size))
        for key,value in header.items():
            if key=="__metadata__":continue
            if key in actual:raise ValueError("Tensor is duplicated across shards: "+key)
            if index.get(key)!=shard:raise ValueError("Shard/index membership mismatch: "+key)
            actual[key]=value["shape"]
        with safe_open(str(path),framework="pt",device="cpu") as tensors:
            for key in tensors.keys():
                if key.startswith(prefixes):
                    value=tensors.get_tensor(key)
                    if not torch.isfinite(value).all():raise ValueError("Non-finite trained navigation tensor: "+key)
                    heads[key]={"shape":list(value.shape),"dtype":str(value.dtype),"finite":True}
    missing=set(expected)-set(actual)
    allowed={"lm_head.weight"} if raw.get("tie_word_embeddings") else set()
    if missing-allowed or set(actual)-set(expected):raise ValueError("Architecture/checkpoint keys differ: "+str((sorted(missing-allowed),sorted(set(actual)-set(expected)))))
    if set(index)!=set(actual):raise ValueError("Some indexed tensors do not exist in the weight shards")
    mismatched=[k for k in actual if actual[k]!=expected[k]]
    if mismatched:raise ValueError("Architecture/checkpoint tensor shapes differ: "+str(mismatched[:8]))
    del model;gc.collect()
    return {"passed":True,"indexed_tensors":len(index),"architecture_keys":len(expected),
            "tied_aliases_omitted":sorted(missing),"trained_heads_checked":heads,"full_weights_loaded":False}


def processor_and_small_forward(checkpoint):
    import numpy as np
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
    from omninav_pointgoal import OmniNavPointGoal,IMAGE_FUNCTIONS,UPSTREAM
    from navigation_sources import load_functions,load_string_assignment
    from PIL import Image
    import math,random
    raw=json.loads((checkpoint/"config.json").read_text())
    raw.update(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=1,
               rope_scaling={"mrope_section":[1,1,2],"rope_type":"default","type":"default"})
    raw["vision_config"].update(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,fullatt_block_indexes=[0])
    config=Qwen2_5_VLConfig.from_dict(raw);config._attn_implementation="eager"
    torch.manual_seed(42);torch.set_num_threads(2)
    model=Qwen2_5_VLForConditionalGeneration(config).eval()
    with torch.no_grad():model.query_action.normal_(0,.02)
    predictor=OmniNavPointGoal.__new__(OmniNavPointGoal)
    predictor.checkpoint=checkpoint
    predictor.model=model;predictor.torch=torch;predictor.expected_arrival_count=5
    predictor.processor=AutoProcessor.from_pretrained(str(checkpoint),min_pixels=56*56,max_pixels=4480*4480,use_fast=False,
                                                     local_files_only=True,trust_remote_code=False)
    token_ids={name:predictor.processor.tokenizer.convert_tokens_to_ids(name)
               for name in ["<input_pos1>","<input_pos2>","<input_pos3>","<input_pos4>","<input_pos5>","<input_target>"]}
    if list(token_ids.values())!=list(range(151665,151671)):raise ValueError("Position-token IDs differ from the native head convention")
    predictor.functions=load_functions(UPSTREAM,IMAGE_FUNCTIONS,{"np":np,"Image":Image,"math":math,"random":random})
    images=[]
    width,height=json.loads((PROJECT/'configs/navigation_camera.json').read_text())['resolution_wh']
    for index in range(7):
        image=np.zeros((height,width,3),dtype=np.uint8)
        image[:]=[25+index*20,100,180-index*10];image[110:250,200:400]=[200,70,40]
        images.append(image)
    request={"capture_time_s":5.,"anchor_pose_xyyaw":[0.,0.,0.],"images":images,
             "input_positions_scaled":[[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,2.]],
             "content_text":load_string_assignment(UPSTREAM,"get_result_fast","content_text")}
    result=predictor.predict(request)
    return {"passed":True,"validation_scope":"random_small_architecture_real_processor_and_original_forward_not_trained_model_inference",
            "image_count":7,"raw_image_resolution_wh":[width,height],"position_token_ids":token_ids,"waypoint_shape":[1,5,2],"arrival_shape":[1,5],
            "latency_s":result["latency_s"],"full_checkpoint_loaded":False,"gpu_used":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=PROJECT/"checkpoints/OmniNav_Slowfast")
    parser.add_argument("--small-forward",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    from model_environment import configure_model_environment
    configure_model_environment(PROJECT)
    checkpoint=args.checkpoint.resolve()
    result={"scope":"offline_CPU_source_and_contract_validation","gpu_used":False,"trained_full_model_inference_run":False}
    result["checkpoint_schema"]=checkpoint_schema(checkpoint)
    if args.small_forward:result["small_forward"]=processor_and_small_forward(checkpoint)
    write_json(args.output,result)
    print(json.dumps({"output":str(args.output),"schema_passed":result["checkpoint_schema"]["passed"],
                      "small_forward_passed":result.get("small_forward",{}).get("passed")},indent=2))


if __name__=="__main__":main()
