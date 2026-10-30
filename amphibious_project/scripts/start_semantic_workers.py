"""Start the two own baseline workers, serializing residency on physical GPU 1.

This helper never starts Isaac, allocates cluster resources, or stops a process.
Idle model weights stay in CPU RAM; the workers exit after ten idle minutes.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from navigation_io import own_path, read_json


PROJECT = Path(__file__).resolve().parents[1]
WORKERS = {
    'waypoint': ('.runtime/navigation/worker', '.venvs/omninav/bin/python', 'scripts/omninav_worker.py',
                 ['--checkpoint', 'checkpoints/OmniNav_Slowfast', '--device', 'cuda:0', '--startup-smoke',
                  '--recover-stale-lock', '--offload-between-requests', '--idle-exit-s', '600']),
    'semantic': ('.runtime/semantic/worker', '.venvs/qwen_semantic/bin/python', 'scripts/semantic_worker.py',
                 ['--idle-exit-s', '600']),
}


def own_venv_python(relative):
    executable = PROJECT/relative
    venv = own_path(PROJECT, executable.parent.parent)
    if not own_path(PROJECT, venv/'pyvenv.cfg').is_file():
        raise RuntimeError('Own model virtual environment is incomplete')
    # A venv intentionally links its interpreter to the installed read-only
    # Isaac Python. Execute through the own venv path to retain its packages.
    expected = Path('/home/ubuntu/qsq/isaacsim/kit/python/bin/python3').resolve()
    if not executable.is_file() or executable.resolve() != expected:
        raise RuntimeError('Unexpected model virtual-environment interpreter')
    return executable


def status(name):
    relative, _, source, _ = WORKERS[name]
    queue = own_path(PROJECT, relative)
    path = queue/'ready.json'
    if not path.exists():
        return {'ready': False, 'name': name}
    ready = read_json(path)
    pid = ready.get('pid')
    proc = Path('/proc', str(pid), 'cmdline')
    argv = proc.read_bytes().split(b'\0') if proc.exists() else []
    exact_source = str(own_path(PROJECT, source)).encode() in argv
    beat = read_json(queue/'heartbeat.json') if (queue/'heartbeat.json').exists() else {}
    live = exact_source and beat.get('worker_id') == ready['worker_id'] and time.time()-beat.get('wall_unix_s', 0) < 6
    stopped = queue/'stopped.json'
    live = live and not (stopped.exists() and read_json(stopped).get('worker_id') == ready['worker_id'])
    return {'name': name, 'ready': bool(live), 'pid': pid, 'worker_id': ready['worker_id'],
            'CPU_offload': ready.get('weights_offloaded_to_CPU_between_requests', False)}


def start(name):
    current = status(name)
    if current['ready']:
        return current
    relative, python, source, arguments = WORKERS[name]
    queue = own_path(PROJECT, relative)
    if (queue/'STOP').exists() or (queue/'worker.lock').exists():
        raise RuntimeError('Inspect the own '+name+' queue lock/STOP before restarting; nothing was removed')
    free = int(subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip())
    if free < 18000:
        raise RuntimeError('GPU 1 currently has less than 18,000 MiB free; no worker was started')
    command = [str(own_venv_python(python)), '-B', str(own_path(PROJECT, source))]+arguments
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    directory = own_path(PROJECT, '.runtime/semantic_services')
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log = directory/(name+'_'+stamp+'.log')
    exit_file = log.with_suffix('.exit.json')
    runner = ('import json,subprocess,time;from pathlib import Path;r=subprocess.run('+repr(command)+
              ',cwd='+repr(str(PROJECT))+');Path('+repr(str(exit_file))+
              ").write_text(json.dumps({'exit_code':r.returncode,'ended_unix_s':time.time()}))")
    with log.open('x') as stream:
        process = subprocess.Popen(['python3', '-B', '-c', runner], cwd=PROJECT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'starting': name, 'supervisor_pid': process.pid, 'log': str(log)}), flush=True)
    limit = time.monotonic()+180
    while time.monotonic() < limit:
        current = status(name)
        if current['ready']:
            return current
        if exit_file.exists():
            raise RuntimeError(name+' startup failed; inspect '+str(log))
        time.sleep(1)
    raise TimeoutError(name+' readiness pending; inspect the own log before another start')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='Start missing own workers; otherwise show status')
    args = parser.parse_args()
    if PROJECT.resolve() != Path('/home/ubuntu/sdc/L17/amphibious_project'):
        raise RuntimeError('Use this helper in the own Linux L17 project')
    for name in WORKERS:
        print(json.dumps(start(name) if args.start else status(name)), flush=True)


if __name__ == '__main__':
    main()
