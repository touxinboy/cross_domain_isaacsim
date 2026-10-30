"""Patch the own request hook and restart exactly the idle own L17 GUI."""
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path('/home/ubuntu/sdc/L17/amphibious_project').resolve(strict=True)
allowed=Path('/home/ubuntu/sdc/L17').resolve(strict=True)
if not root.is_relative_to(allowed):raise RuntimeError('Own L17 project is required')
request=root/'.runtime/robot_ui/rgbd_restart_status.request.json'
response=request.with_name('rgbd_restart_status.result.json')
if response.exists():response.unlink()
request.write_text(json.dumps({'action':'status'}))
for _ in range(20):
    if response.is_file():break
    time.sleep(.15)
if not response.is_file():raise RuntimeError('Own GUI did not respond; no process was stopped')
status=json.loads(response.read_text())
state=status['result']
commands=(state.get('runtime') or {}).get('commands') or {}
if state['validation_running'] or any(abs(float(v))>1e-8 for key in ['tracks','thrusters'] for v in commands.get(key,[])):
    raise RuntimeError('Own GUI is active; no process was stopped')
ui=root/'scripts/robot_task_ui.py'
text=ui.read_text()
anchor='                    elif action=="capture":result={"file":await self.capture(data["filename"])}\n'
old_addition=('                    elif action=="rgbd_capture":\n'
          '                        if self.busy():raise RuntimeError("Component validation is running")\n'
          '                        from rgbd_sensor_rig import get_rig\n'
          '                        result=await get_rig(self.project).capture_all(run_calibration=data.get("calibration",False))\n')
addition=('                    elif action=="rgbd_capture":\n'
          '                        if self.busy():raise RuntimeError("Component validation is running")\n'
          '                        import rgbd_sensor_rig\n'
          '                        if data.get("reload",False):\n'
          '                            import importlib\n'
          '                            rgbd_sensor_rig.dispose_rig();importlib.reload(rgbd_sensor_rig)\n'
          '                        result=await rgbd_sensor_rig.get_rig(self.project).capture_all(\n'
          '                            run_calibration=data.get("calibration",False),include_reference=data.get("include_reference",False))\n')
backup=root/'.runtime/rgbd_backups'
backup.mkdir(exist_ok=True)
if 'action=="rgbd_capture"' not in text:
    if old_addition in text:patched=text.replace(old_addition,addition)
    elif 'action=="rgbd_capture"' not in text and text.count(anchor)==1:patched=text.replace(anchor,anchor+addition)
    else:raise RuntimeError('Own UI source differs; inspect before changing')
    ast.parse(patched)
    (backup/('robot_task_ui_'+hashlib.sha256(text.encode()).hexdigest()[:12]+'.py')).write_text(text)
    ui.write_text(patched)
for name in ['depth_localization.py','localize_rgbd_target.py','rgbd_sensor_rig.py','start_rgbd_validation.py']:
    ast.parse((root/'scripts'/name).read_text())
matches=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
        if argv and Path(argv[0]).name=='kit' and '--portable-root' in argv and argv[argv.index('--portable-root')+1]==str(root/'.runtime/kit'):
            matches.append(int(proc.name))
    except OSError:pass
if len(matches)!=1:raise RuntimeError('Expected exactly one own Kit process; none were stopped')
pid=matches[0]
os.kill(pid,signal.SIGTERM)
for _ in range(80):
    if not Path('/proc',str(pid)).exists():break
    time.sleep(.2)
if Path('/proc',str(pid)).exists():raise RuntimeError('Own Kit did not exit; refusing a duplicate instance')
out=root/'results/rgbd';out.mkdir(parents=True,exist_ok=True)
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
log=out/('launch_'+stamp+'.log')
with log.open('x') as stream:
    process=subprocess.Popen(['python3','-B',str(root/'scripts/launch.py'),'gui','--local-assets-only',
                              '--physics-gpu-index','0','--exec-script','scripts/start_rgbd_validation.py'],
                             cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'stopped_own_kit_pid':pid,'own_launcher_pid':process.pid,'log':str(log),
         'gpu_index':0,'other_processes_modified':False,'created':stamp,'scene_before':state['scene']}
(root/'.runtime/rgbd_gui_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
