"""Send only an appearance-described semantic task to the own running Isaac GUI."""
import argparse
from datetime import datetime
from pathlib import Path
import time
import uuid

from navigation_io import own_path,read_json,write_json

PROJECT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',choices=['flat_top_jacket'],default='flat_top_jacket')
    parser.add_argument('--send',action='store_true');parser.add_argument('--status',action='store_true');parser.add_argument('--stop',action='store_true')
    args=parser.parse_args();current=PROJECT/'.runtime/semantic/current_request.json'
    if args.status or args.stop:
        request=read_json(current);output=own_path(PROJECT,request['output_relative'])
        if args.stop:write_json(own_path(PROJECT,request['stop_relative']),{'reason':'user_stop'})
        else:
            path=output/('summary.json' if (output/'summary.json').exists() else 'progress.json')
            print(path.read_text() if path.exists() else 'Starting semantic task; see own worker and GUI logs.')
        return
    config=read_json(PROJECT/'configs/semantic_navigation.json')
    run_id='run_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    request={'schema':'L17.semantic_navigation_run.v1','run_id':run_id,'mode':'execute','target_id':args.target,
             'instruction':config['instruction'],'start_deadline_wall_unix_s':time.time()+300,
             'output_relative':'results/semantic/'+run_id,'stop_relative':'.runtime/semantic/'+run_id+'.stop.json'}
    write_json(current,request);print('Prepared appearance-only semantic task:',run_id)
    if not args.send:return
    queue=PROJECT/'.runtime/robot_ui';name='semantic_'+uuid.uuid4().hex
    write_json(queue/(name+'.request.json'),{'action':'navigation_start','request':request})
    result=queue/(name+'.result.json');limit=time.monotonic()+15
    while time.monotonic()<limit:
        if result.exists():
            reply=read_json(result)
            if not reply.get('ok'):raise RuntimeError(reply.get('error','Own GUI rejected task'))
            print('Own GUI accepted semantic task:',run_id);return
        time.sleep(.1)
    raise RuntimeError('Own GUI did not accept the semantic request; use the updated own panel')


if __name__=='__main__':main()
