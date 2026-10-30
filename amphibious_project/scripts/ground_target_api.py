"""Obtain a visible target's 2D box and solid-surface pixel from Qwen-VL.

This returns image measurements, never hidden Stage positions or a driving goal.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from PIL import Image

from qwen_api_client import QwenClient


def parse_grounding(raw, width, height):
    text=raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text='\n'.join(text.splitlines()[1:-1])
    data=json.loads(text)
    if not isinstance(data,dict) or type(data.get('visible')) is not bool:
        raise ValueError('Grounding must contain a boolean visible field')
    if not data['visible']:
        return {'visible':False,'point_uv':None,'bbox_xyxy':None,'evidence':str(data.get('evidence',''))}
    def vector(name,size):
        value=data.get(name)
        if not isinstance(value,list) or len(value)!=size or any(type(x) not in [int,float] or not math.isfinite(x) or not 0<=x<=1000 for x in value):
            raise ValueError('Invalid normalized '+name)
        return value
    box=vector('bbox_2d',4)
    point=vector('point_2d',2)
    if not box[0]<box[2] or not box[1]<box[3] or not box[0]<=point[0]<=box[2] or not box[1]<=point[1]<=box[3]:
        raise ValueError('Grounded point must be inside a nonempty box')
    return {'visible':True,'bbox_xyxy':[box[0]*width/1000,box[1]*height/1000,box[2]*width/1000,box[3]*height/1000],
            'point_uv':[min(width-1,point[0]*width/1000),min(height-1,point[1]*height/1000)],
            'bbox_normalized_1000':box,'point_normalized_1000':point,'evidence':str(data.get('evidence',''))}


def ground(client,image_path,target):
    path=Path(image_path).resolve()
    with Image.open(path) as image: width,height=image.size
    prompt=(f'Locate this target in the image: {target}\n'
            'Return only a JSON object: {"visible":true,"bbox_2d":[left,top,right,bottom],'
            '"point_2d":[u,v],"evidence":"brief visual evidence"}. '
            'All coordinates are normalized to 0..1000 over the full image. '
            'Choose point_2d on a visible opaque solid part of the target, preferably on a lower structural member. '
            'For lattice frameworks, avoid empty gaps, water, shadows and background between bars. '
            'The point must be inside the box. If the target is not visible, use visible=false and null coordinates. '
            'Do not provide world coordinates.')
    raw=client.describe([path],prompt,max_tokens=384)
    result={'image':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'image_size_wh':[width,height],'target_description':target,'model':client.model,
            'raw_response':raw,'latency_s':client.last_latency_s,'usage':client.last_usage,
            'provider':'qwen_visual_grounding','is_navigation_goal':False}
    try:
        result.update(parse_grounding(raw,width,height))
        result['format_valid']=True
    except (ValueError,TypeError) as error:
        result.update(format_valid=False,error=str(error),visible=False,point_uv=None,bbox_xyxy=None)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--target',required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=ground(QwenClient(timeout_s=120.),args.image,args.target)
    output=args.output or args.image.with_suffix('.grounding.json')
    if output.resolve()==args.image.resolve(): raise ValueError('Do not overwrite the source image')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['format_valid'] else 3


if __name__=='__main__': raise SystemExit(main())
