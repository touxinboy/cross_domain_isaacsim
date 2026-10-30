"""Ask the existing own Isaac GUI to save calibrated RGB-D; starts no simulator."""
import argparse
import json
from pathlib import Path
import time
import uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',action='store_true')
    parser.add_argument('--reference-view',action='store_true',help='Diagnostic fixed camera, excluded by default')
    parser.add_argument('--reload-sensors',action='store_true')
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    allowed=Path('/home/ubuntu/sdc/L17').resolve(strict=True)
    if not project.is_relative_to(allowed):raise RuntimeError('Use the own L17 project on the Isaac machine')
    directory=project/'.runtime/robot_ui'
    if not directory.is_dir():raise RuntimeError('Open the own robot scene first')
    name='rgbd_'+uuid.uuid4().hex
    pending=directory/(name+'.tmp')
    request=directory/(name+'.request.json')
    result=directory/(name+'.result.json')
    pending.write_text(json.dumps({'action':'rgbd_capture','calibration':args.calibration,
                                  'reload':args.reload_sensors,'include_reference':args.reference_view}))
    pending.replace(request)
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        if result.is_file():
            response=json.loads(result.read_text())
            print(json.dumps(response,ensure_ascii=False,indent=2))
            return 0 if response.get('ok') else 3
        time.sleep(.2)
    raise RuntimeError('No GUI response within 45 seconds; no new simulator or GPU job was started')


if __name__=='__main__':raise SystemExit(main())
