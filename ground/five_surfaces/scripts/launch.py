"""Run the five-surface project with bwrap or Landlock file write isolation."""

import argparse
import json
from importlib.metadata import distribution, PackageNotFoundError
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path(__file__).resolve().parents[2]


def isaac_entrypoint():
    try:
        package = distribution("isaacsim")
    except PackageNotFoundError as error:
        raise RuntimeError("Activate conda environment isaacsim4.5 before launching.") from error
    if package.version not in ("4.5.0", "4.5.0.0"):
        raise RuntimeError(f"This project requires Isaac Sim 4.5.0; found {package.version}.")
    entrypoint = Path(sys.executable).with_name("isaacsim")
    if not entrypoint.is_file():
        raise RuntimeError(f"Isaac Sim entrypoint is missing: {entrypoint}")
    return str(entrypoint)


def safe_directory(path, root):
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"Write path escapes ground workspace: {path}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def select_isolation(requested, root):
    failure = 'bwrap is not installed'
    if requested != 'landlock' and Path('/usr/bin/bwrap').is_file():
        probe = subprocess.run(['/usr/bin/bwrap', '--ro-bind', '/', '/', '/usr/bin/true'],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            return 'bwrap', None
        failure = probe.stderr.strip() or f'bwrap returned {probe.returncode}'
    if requested == 'bwrap':
        raise RuntimeError(f'bwrap is unavailable: {failure}')
    helper = PROJECT / 'scripts/landlock_exec.py'
    probe = subprocess.run([sys.executable, '-B', str(helper), '--root', str(root), '--probe'],
                           capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError(f'No supported file write isolation is available. bwrap: {failure}\n'
                           f'Landlock: {probe.stderr.strip()}')
    return 'landlock', failure if requested == 'auto' else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gui", "check", "turning", "straight"))
    parser.add_argument("--exec-script", help="Optional one-time GUI script inside the ground workspace.")
    parser.add_argument("--headless", action="store_true", help="Run the app without a desktop window.")
    parser.add_argument('--isolation', choices=('auto', 'bwrap', 'landlock'), default='auto',
                        help='Prefer bwrap; use Landlock when user namespaces are blocked.')
    parser.add_argument("--physics-gpu-index", type=int, choices=(0, 1),
                        help="Optional GPU index for this own app; default preserves task.json.")
    parser.add_argument("--local-assets-only", action="store_true",
                        help="Use an own local asset catalog; skip unavailable external Nucleus lookups.")
    args = parser.parse_args()
    root = EXPECTED_ROOT.resolve(strict=True)
    if root != EXPECTED_ROOT or not PROJECT.is_relative_to(root):
        raise RuntimeError("This launcher must be inside the verified ground directory.")
    isolation, fallback_reason = select_isolation(args.isolation, root)
    config = json.loads((PROJECT / "configs/task.json").read_text())
    runtime = safe_directory(PROJECT / ".runtime", root)
    names = ("applications", "cache", "config", "data", "state", "run", "tmp", "shm", "logs", "kit", "cuda", "gl", "optix", "screenshots", "torch", "torch_extensions", "warp", "mpl", "numba", "huggingface")
    paths = {name: safe_directory(runtime / name, root) for name in names}
    paths["run"].chmod(0o700)
    if args.mode in ("gui", "turning", "straight"):
        entrypoint = isaac_entrypoint()
        # Kit normally saves acceptance beside its package, which is read-only here.
        # Keep a user's interactive acceptance in this project's own config instead.
        eula_record = paths["config"] / "kit_eula_accepted"
        accepted = os.environ.get("OMNI_KIT_ACCEPT_EULA", "").lower() in ("yes", "y", "1")
        if not accepted and not eula_record.is_file():
            print("Isaac Sim requires acceptance of the NVIDIA Omniverse license:", flush=True)
            print("https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html", flush=True)
            if input("Do you accept the EULA? (Yes/No): ").strip().lower() not in ("yes", "y"):
                raise RuntimeError("NVIDIA license was not accepted.")
            eula_record.write_text("yes\n")
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment = {
        "XDG_CACHE_HOME": str(paths["cache"]), "XDG_CONFIG_HOME": str(paths["config"]),
        "XDG_DATA_HOME": str(paths["data"]), "XDG_STATE_HOME": str(paths["state"]),
        "XDG_RUNTIME_DIR": str(paths["run"]), "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_CACHE_PATH": str(paths["cuda"]),
        "__GL_SHADER_DISK_CACHE_PATH": str(paths["gl"]), "TORCH_HOME": str(paths["torch"]),
        "OPTIX_CACHE_PATH": str(paths["optix"]),
        "TORCH_EXTENSIONS_DIR": str(paths["torch_extensions"]), "WARP_CACHE_PATH": str(paths["warp"]),
        "MPLCONFIGDIR": str(paths["mpl"]), "NUMBA_CACHE_DIR": str(paths["numba"]),
        "HF_HOME": str(paths["huggingface"]), "GIO_USE_VFS": "local",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={paths['run']}/unused-bus",
    }
    command = ["/usr/bin/bwrap", "--die-with-parent", "--new-session", "--ro-bind", "/", "/",
               "--proc", "/proc", "--dev", "/dev",
               "--bind", str(paths["tmp"]), "/tmp", "--bind", str(paths["shm"]), "/dev/shm",
               "--bind", str(root), str(root),
               "--chdir", str(PROJECT)]
    # Redirect Kit's desktop shortcut into the project without changing the host's files.
    desktop_apps = Path.home() / ".local/share/applications"
    if desktop_apps.is_dir():
        command.extend(["--bind", str(paths["applications"]), str(desktop_apps)])
    devices = list(Path("/dev").glob("nvidia*"))
    if Path("/dev/dri").is_dir():
        devices.append(Path("/dev/dri"))
    for device in devices:
        command.extend(["--dev-bind", str(device), str(device)])
    if args.mode == "gui" and not args.headless:
        sockets = sorted(Path("/tmp/.X11-unix").glob("X*"))
        display = os.environ.get("DISPLAY") or (":" + sockets[0].name[1:] if sockets else None)
        if not display:
            raise RuntimeError("No X11 desktop was found; connect ToDesk and try again.")
        environment["DISPLAY"] = display
        auth = os.environ.get("XAUTHORITY")
        if not auth:
            auth = next((str(p) for p in (Path(f"/run/user/{os.getuid()}/gdm/Xauthority"), Path.home() / ".Xauthority") if p.is_file()), None)
        if auth:
            environment["XAUTHORITY"] = auth
        command.extend(["--ro-bind", "/tmp/.X11-unix", "/tmp/.X11-unix"])
    for key, value in environment.items():
        command.extend(["--setenv", key, value])
    if args.mode in ("gui", "turning", "straight"):
        default_script = {'turning':'scripts/run_turning.py', 'straight':'scripts/run_straight.py'}.get(args.mode, 'scripts/open_stage.py')
        startup = (PROJECT / args.exec_script).resolve() if args.exec_script else PROJECT / default_script
        if not startup.is_relative_to(root) or not startup.is_file():
            raise RuntimeError("The GUI startup script must be an existing file inside the ground workspace.")
        experience = str(PROJECT/'configs/turning_physics.kit') if args.mode in ('turning', 'straight') else 'isaacsim.exp.full.kit'
        target = [entrypoint, experience, "--portable-root", str(paths["kit"]),
                  "--/plugins/carb.tasking.plugin/threadCount=4", "--/plugins/omni.tbb.globalcontrol/maxThreadCount=4",
                  "--/app/renderer/resolution/width=1280", "--/app/renderer/resolution/height=720",
                  f"--/renderer/activeGpu={config.get('gui_gpu_index', 0)}", "--/renderer/multiGpu/enabled=false",
                  f"--/physics/cudaDevice={args.physics_gpu_index if args.physics_gpu_index is not None else config['gpu_index']}", "--/app/extensions/registryEnabled=false",
                  "--/isaac/startup/create_new_stage=false", "--/isaac/startup/ros_bridge_extension=",
                  f"--/persistent/app/captureFrame/path={paths['screenshots']}/",
                  "--/app/window/title=Ground - Five Surfaces", "--exec", str(startup)]
        if args.mode in ('turning', 'straight'):
            package = Path(distribution('isaacsim').locate_file('isaacsim'))
            for folder in ('exts', 'extscache', 'extsPhysics', 'extsDeprecated'):
                target.extend(['--ext-folder', str(package/folder)])
            environment['GROUND_TURN_PHYSICS_ONLY'] = '1'
        if args.headless or args.mode in ('turning', 'straight'):
            target.extend(["--no-window", "--/app/file/ignoreUnsavedOnExit=true"])
        if args.local_assets_only:
            local_assets = safe_directory(runtime / "local_assets", root)
            for catalog in ("Isaac", "NVIDIA"):
                safe_directory(local_assets / catalog, root)
            target.append(f"--/persistent/isaac/asset_root/default={local_assets}")
    else:
        # os.access() does not account for Landlock. Actually attempt an unnamed
        # temporary file, which must fail outside ground, and allow an own one.
        probe = ("import errno,os,sys,tempfile\n"
                 "with tempfile.TemporaryFile(dir='.runtime/tmp') as f: f.write(b'own workspace')\n"
                 "try:\n"
                 "    f=tempfile.TemporaryFile(dir=sys.prefix)\n"
                 "except OSError as error:\n"
                 "    if error.errno not in (errno.EACCES,errno.EPERM,errno.EROFS): raise\n"
                 "    print('GROUND_ISOLATION_OK'); print('project='+os.getcwd()); print('outside_writable=False')\n"
                 "else:\n"
                 "    f.close(); raise RuntimeError('Outside write restriction failed')\n")
        target = [sys.executable, "-B", "-c", probe]
    if isolation == 'landlock':
        environment['TMPDIR'] = str(paths['tmp'])
        command = [sys.executable, '-B', str(PROJECT / 'scripts/landlock_exec.py'),
                   '--root', str(root), '--applications', str(paths['applications']), '--']
    child_environment = os.environ.copy()
    child_environment.update(environment)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log = paths["logs"] / f"{args.mode}_{stamp}.log"
    print(f"Mode: {args.mode}\nLog: {log}", flush=True)
    print(f'Isolation: {isolation}', flush=True)
    if fallback_reason:
        print(f'bwrap unavailable ({fallback_reason}); using Landlock file write isolation.', flush=True)
    with log.open("x") as handle:
        handle.write(f'Isolation: {isolation}\n')
        handle.flush()
        completed = subprocess.run(command + target, stdout=handle, stderr=subprocess.STDOUT,
                                   env=child_environment, cwd=PROJECT)
    print(log.read_text(errors="replace")[-8000:], flush=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
