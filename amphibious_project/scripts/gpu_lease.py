"""Serialize only this project's model GPU residency; never manage other jobs."""
from contextlib import contextmanager
from pathlib import Path
import time
import gc


@contextmanager
def own_gpu_lease(project,timeout_s=180.):
    import fcntl
    project=Path(project).resolve()
    if not project.is_relative_to(Path('/home/ubuntu/sdc/L17')):raise RuntimeError('GPU lease belongs to the own L17 project')
    path=(project/'.runtime/model_gpu_lease.lock').resolve()
    if not path.is_relative_to(project):raise RuntimeError('GPU lease symlink escapes project')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        limit=time.monotonic()+timeout_s
        while True:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>limit:raise TimeoutError('Own model GPU lease timed out')
                time.sleep(.05)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def offload_model(backend):
    backend.model.to('cpu')
    gc.collect()
    backend.torch.cuda.synchronize()
    backend.torch.cuda.empty_cache()


def residency_state(backend):
    return {'parameter_devices':sorted({str(p.device) for p in backend.model.parameters()}),
            'buffer_devices':sorted({str(p.device) for p in backend.model.buffers()}),
            'cuda_allocated_bytes':backend.torch.cuda.memory_allocated(),
            'cuda_reserved_bytes':backend.torch.cuda.memory_reserved()}


@contextmanager
def resident_model(project,backend,device='cuda:0'):
    with own_gpu_lease(project):
        needed=sum(p.numel()*p.element_size() for p in backend.model.parameters())
        with backend.torch.cuda.device(device):free,_=backend.torch.cuda.mem_get_info()
        if free<needed+2*1024**3:raise RuntimeError('Selected model GPU lacks free memory plus reserve')
        backend.model.to(device)
        try:yield
        finally:offload_model(backend)


class OffloadedPredictor:
    def __init__(self,project,backend,device):self.project=project;self.backend=backend;self.device=device
    def __getattr__(self,name):return getattr(self.backend,name)
    def predict(self,request):
        with resident_model(self.project,self.backend,self.device):return self.backend.predict(request)
