"""Combine an exact RGB-frame grounding with its calibrated depth observation."""
import argparse
import json
from pathlib import Path

from depth_localization import DepthFrame,localize_surface


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame',type=Path,required=True)
    parser.add_argument('--grounding',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=localize_surface(DepthFrame.load(args.frame),json.loads(args.grounding.read_text(encoding='utf-8')))
    output=args.output or args.frame.with_name(args.frame.stem+'.surface.json')
    if output.resolve() in [args.frame.resolve(),args.grounding.resolve()]: raise ValueError('Preserve input observations')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['accepted'] else 3


if __name__=='__main__':raise SystemExit(main())
