"""Own local-file Qwen3-VL/MSGNav + online RGB-D planning worker; no listener."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

from model_environment import configure_model_environment
from navigation_io import file_sha, own_path, read_json, write_json
from qwen_api_client import QwenClient
from semantic_agent import SemanticPlanner
from gpu_lease import own_gpu_lease,offload_model,resident_model


PROJECT=Path(__file__).resolve().parents[1]


class InProcessQwenClient(QwenClient):
    def __init__(self,backend,normalizer,project):
        super().__init__('http://127.0.0.1/v1',model='l17-qwen3-vl-4b')
        self.backend=backend;self.normalize=normalizer;self.completed_calls=0
        self.project=project

    def _request(self,method,endpoint,payload=None):
        if method!='POST' or endpoint!='/chat/completions':raise ValueError('This client uses only in-process model calls')
        started=time.monotonic()
        try:
            with resident_model(self.project,self.backend):
                text,usage=self.backend.generate(self.normalize(payload['messages']),payload['max_tokens'],payload['temperature'])
            self.completed_calls+=1
            return {'choices':[{'message':{'content':text}}],'usage':usage}
        finally:self.last_latency_s=time.monotonic()-started


def code_identity(project):
    names=['scripts/semantic_worker.py','scripts/semantic_agent.py','scripts/online_terrain_map.py',
           'configs/navigation_camera.json','deploy/qwen_a100/qwen_transformers_server.py']
    return {n:file_sha(project/n) for n in names}


def process_request(project,path,agent,worker_id):
    request_id=path.name.removesuffix('.request.json');reply={'schema':'L17.semantic_result.v1','request_id':request_id,
        'worker_id':worker_id,'controller_epoch':None,'observation_sha256':None,'ok':False}
    started=time.monotonic()
    try:
        request=read_json(path)
        if request.get('schema')!='L17.semantic_request.v1' or request['request_id']!=request_id:raise ValueError('Invalid semantic request identity')
        for k in ['controller_epoch','observation_sha256']:reply[k]=request[k]
        if request.get('worker_id')!=worker_id:raise ValueError('Semantic request belongs to another worker instance')
        if time.time()>float(request['deadline_wall_unix_s']):raise ValueError('Semantic request expired')
        if path.with_name(request_id+'.cancelled.json').exists():raise ValueError('Semantic request cancelled')
        if file_sha(own_path(project,request['observation_file']))!=request['observation_sha256']:raise ValueError('Semantic observation modified')
        reply['result']=agent.update(request);reply['ok']=True
    except Exception:reply['error']=traceback.format_exc()
    reply['wall_s']=time.monotonic()-started
    write_json(path.with_name(request_id+'.result.json'),reply)
    return reply


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='checkpoints/Qwen3-VL-4B-Instruct')
    parser.add_argument('--idle-exit-s',type=float,default=1200)
    args=parser.parse_args();project=PROJECT.resolve()
    if not project.is_relative_to(Path('/home/ubuntu/sdc/L17')):raise RuntimeError('Worker must stay in own L17')
    configure_model_environment(project)
    queue=own_path(project,'.runtime/semantic/worker');queue.mkdir(parents=True,exist_ok=True)
    if (queue/'STOP').exists():raise RuntimeError('Own semantic STOP marker exists')
    lock=queue/'worker.lock';worker_id=uuid.uuid4().hex
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(descriptor,json.dumps({'pid':os.getpid(),'worker_id':worker_id}).encode());os.close(descriptor)
    try:
        (queue/'ready.json').unlink(missing_ok=True)
        source=project/'deploy/qwen_a100/qwen_transformers_server.py'
        spec=importlib.util.spec_from_file_location('l17_qwen_backend',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        checkpoint=own_path(project,args.checkpoint);preflight=module.preflight(checkpoint)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one selected semantic GPU')
        with own_gpu_lease(project):
            free,_=torch.cuda.mem_get_info()
            if free<preflight['weight_bytes']+3*1024**3:raise RuntimeError('Insufficient free memory for the semantic model and reserve')
            backend=module.TransformersBackend(str(checkpoint))
            offload_model(backend)
        backend.torch.manual_seed(42)
        client=InProcessQwenClient(backend,module.normalize_messages,project)
        smoke=client.chat([{'role':'user','content':'Reply with the single word READY.'}],max_tokens=16)
        if 'READY' not in smoke.upper():raise RuntimeError('Qwen text forward did not pass startup check')
        agent=SemanticPlanner(project,client)
        ready={'worker_id':worker_id,'pid':os.getpid(),'project':str(project),'code_sha256':code_identity(project),
               'model_loaded':True,'model':'Qwen3-VL-4B-Instruct','model_preflight':preflight,'startup_forward_passed':True}
        ready['weights_offloaded_to_CPU_between_requests']=True
        write_json(queue/'ready.json',ready)
        write_json(queue/'heartbeat.json',{'worker_id':worker_id,'wall_unix_s':time.time(),'completed_VLM_calls':client.completed_calls})
        print('L17_SEMANTIC_WORKER_READY',json.dumps({'worker_id':worker_id,'pid':os.getpid()}),flush=True)
        last_work=last_beat=time.monotonic()
        while time.monotonic()-last_work<args.idle_exit_s:
            if (queue/'STOP').exists():break
            for path in sorted(queue.glob('*.request.json')):
                if path.with_name(path.name.replace('.request.json','.result.json')).exists():continue
                process_request(project,path,agent,worker_id);last_work=time.monotonic()
            if time.monotonic()-last_beat>1:
                write_json(queue/'heartbeat.json',{'worker_id':worker_id,'wall_unix_s':time.time(),'completed_VLM_calls':client.completed_calls})
                last_beat=time.monotonic()
            time.sleep(.05)
    except BaseException:
        write_json(queue/'startup_or_runtime_error.json',{'worker_id':worker_id,'error':traceback.format_exc()});raise
    finally:
        lock.unlink(missing_ok=True)
        write_json(queue/'stopped.json',{'worker_id':worker_id,'pid':os.getpid(),'wall_unix_s':time.time()})


if __name__=='__main__':main()
