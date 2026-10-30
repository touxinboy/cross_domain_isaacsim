"""Run only this project's workflows with all other persistent paths read-only."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ubuntu/sdc/L17")
ISAAC = Path("/home/ubuntu/qsq/isaacsim")


def safe_directory(path, root):
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"Write path escapes L17: {path}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "gui", "check"))
    parser.add_argument("--exec-script", help="Optional one-time GUI script inside L17.")
    parser.add_argument("--physics-gpu-index", type=int, choices=(0, 1),
                        help="Optional GPU index for this own app; default preserves task.json.")
    parser.add_argument("--local-assets-only", action="store_true",
                        help="Use an own local asset catalog; skip unavailable external Nucleus lookups.")
    args = parser.parse_args()
    root = EXPECTED_ROOT.resolve(strict=True)
    if root != EXPECTED_ROOT or not PROJECT.is_relative_to(root):
        raise RuntimeError("This launcher must be inside the verified L17 directory.")
    if not Path("/usr/bin/bwrap").is_file():
        raise RuntimeError("bwrap is required: refusing an unisolated launch.")
    config = json.loads((PROJECT / "configs/task.json").read_text())
    runtime = safe_directory(PROJECT / ".runtime", root)
    names = ("cache", "config", "data", "state", "run", "tmp", "shm", "logs", "kit", "cuda", "gl", "torch", "torch_extensions", "warp", "mpl", "numba", "huggingface")
    paths = {name: safe_directory(runtime / name, root) for name in names}
    paths["run"].chmod(0o700)
    environment = {
        "XDG_CACHE_HOME": str(paths["cache"]), "XDG_CONFIG_HOME": str(paths["config"]),
        "XDG_DATA_HOME": str(paths["data"]), "XDG_STATE_HOME": str(paths["state"]),
        "XDG_RUNTIME_DIR": str(paths["run"]), "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_CACHE_PATH": str(paths["cuda"]),
        "__GL_SHADER_DISK_CACHE_PATH": str(paths["gl"]), "TORCH_HOME": str(paths["torch"]),
        "TORCH_EXTENSIONS_DIR": str(paths["torch_extensions"]), "WARP_CACHE_PATH": str(paths["warp"]),
        "MPLCONFIGDIR": str(paths["mpl"]), "NUMBA_CACHE_DIR": str(paths["numba"]),
        "HF_HOME": str(paths["huggingface"]), "GIO_USE_VFS": "local",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={paths['run']}/unused-bus",
    }
    command = ["/usr/bin/bwrap", "--die-with-parent", "--new-session", "--ro-bind", "/", "/",
               "--bind", str(root), str(root), "--proc", "/proc", "--dev", "/dev",
               "--bind", str(paths["tmp"]), "/tmp", "--bind", str(paths["shm"]), "/dev/shm",
               "--chdir", str(PROJECT)]
    devices = list(Path("/dev").glob("nvidia*"))
    if Path("/dev/dri").is_dir():
        devices.append(Path("/dev/dri"))
    for device in devices:
        command.extend(["--dev-bind", str(device), str(device)])
    if args.mode == "gui":
        sockets = sorted(Path("/tmp/.X11-unix").glob("X*"))
        display = os.environ.get("DISPLAY") or (":" + sockets[0].name[1:] if sockets else None)
        if not display:
            raise RuntimeError("No X11 desktop was found; connect ToDesk and try again.")
        environment["DISPLAY"] = display
        auth = os.environ.get("XAUTHORITY")
        if not auth:
            auth = next((str(p) for p in (Path("/run/user/1000/gdm/Xauthority"), Path("/home/ubuntu/.Xauthority")) if p.is_file()), None)
        if auth:
            environment["XAUTHORITY"] = auth
        command.extend(["--ro-bind", "/tmp/.X11-unix", "/tmp/.X11-unix"])
    for key, value in environment.items():
        command.extend(["--setenv", key, value])
    if args.mode == "prepare":
        target = ["/bin/bash", str(ISAAC / "python.sh"), str(PROJECT / "scripts/new_scene.py")]
    elif args.mode == "gui":
        startup = (PROJECT / args.exec_script).resolve() if args.exec_script else PROJECT / "scripts/open_stage.py"
        if not startup.is_relative_to(root) or not startup.is_file():
            raise RuntimeError("The GUI startup script must be an existing file inside L17.")
        target = ["/bin/bash", str(ISAAC / "isaac-sim.sh"), "--portable-root", str(paths["kit"]),
                  f"--/renderer/activeGpu={config.get('gui_gpu_index', 0)}", "--/renderer/multiGpu/enabled=false",
                  f"--/physics/cudaDevice={args.physics_gpu_index if args.physics_gpu_index is not None else config['gpu_index']}", "--/app/extensions/registryEnabled=false",
                  "--/app/window/title=L17 - Amphibious Navigation", "--exec", str(startup)]
        if args.local_assets_only:
            local_assets = safe_directory(runtime / "local_assets", root)
            for catalog in ("Isaac", "NVIDIA"):
                safe_directory(local_assets / catalog, root)
            target.append(f"--/persistent/isaac/asset_root/default={local_assets}")
    else:
        probe = "import os,pathlib; print('L17_ISOLATION_OK'); print('project='+os.getcwd()); print('outside_writable='+str(os.access('/home/ubuntu/qsq/isaacsim',os.W_OK)))"
        target = ["/usr/bin/python3", "-B", "-c", probe]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log = paths["logs"] / f"{args.mode}_{stamp}.log"
    print(f"Mode: {args.mode}\nLog: {log}", flush=True)
    with log.open("x") as handle:
        completed = subprocess.run(command + target, stdout=handle, stderr=subprocess.STDOUT)
    print(log.read_text(errors="replace")[-8000:], flush=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
