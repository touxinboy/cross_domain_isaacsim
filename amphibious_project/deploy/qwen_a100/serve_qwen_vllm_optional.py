"""Optional older vLLM/Qwen2.5 template; not the selected Qwen3-VL route.

Use --check-only without a GPU. --start requires a Slurm allocation and writes
only below the current account's ~/L17/qwen_service. Existing jobs are untouched.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import sys


def check():
    value = os.environ.get("QWEN_MODEL_DIR", "")
    if not value:
        raise ValueError("Set QWEN_MODEL_DIR to an existing local Qwen2.5-VL-7B-Instruct checkpoint")
    model = Path(value).expanduser().resolve()
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2_5_vl" or "Qwen2_5_VLForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("This template expects Qwen2.5-VL, not plain Qwen or a navigation-head checkpoint")
    if not any(path.is_file() and path.stat().st_size > 0 for path in model.glob("*.safetensors")):
        raise ValueError("No local safetensors model weights found; this script does not download them")
    version = importlib.metadata.version("vllm")
    port = int(os.environ.get("QWEN_PORT", "18080"))
    if not 1024 <= port <= 65535:
        raise ValueError("QWEN_PORT must be between 1024 and 65535")
    own_l17 = (Path.home() / "L17").resolve()
    root = Path(os.environ.get("L17_QWEN_ROOT", str(own_l17 / "qwen_service"))).expanduser().resolve()
    if not root.is_relative_to(own_l17) or root == own_l17:
        raise ValueError("Keep all service runtime files in a subdirectory of ~/L17")
    return {"model_dir": str(model), "vllm_version": version, "runtime_root": str(root),
            "listen_host": "127.0.0.1", "port": port,
            "served_model_name": "l17-qwen2.5-vl-7b", "slurm_allocation_present": bool(os.environ.get("SLURM_JOB_ID"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--start", action="store_true")
    args = parser.parse_args()
    status = check()
    if args.check_only:
        print(json.dumps(status, indent=2))
        return 0
    if not status["slurm_allocation_present"] or os.environ.get("CUDA_VISIBLE_DEVICES", "") in ["", "-1", "NoDevFiles"]:
        raise RuntimeError("Start through a separate single-GPU Slurm job, not the SSH login shell")
    root = Path(status["runtime_root"])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.umask(0o077)
    paths = {"HF_HOME": "cache/huggingface", "XDG_CACHE_HOME": "cache", "TMPDIR": "tmp",
             "VLLM_CACHE_ROOT": "cache/vllm", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
             "PYTHONPYCACHEPREFIX": "cache/pycache"}
    for variable, relative in paths.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.environ[variable] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not os.environ.get("VLLM_API_KEY"):
        credentials = root / "credentials"
        credentials.mkdir(exist_ok=True, mode=0o700)
        credentials.chmod(0o700)
        key_file = credentials / "api_key"
        if key_file.is_symlink():
            raise RuntimeError("Use a regular private API key file in the own service directory")
        if not key_file.exists():
            with key_file.open("x", encoding="utf-8") as stream:
                stream.write(secrets.token_urlsafe(32) + "\n")
        key_file.chmod(0o600)
        os.environ["VLLM_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
    if not os.environ["VLLM_API_KEY"]:
        raise ValueError("API key must not be empty")
    # The key stays in the environment and is not included in printed arguments.
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", status["model_dir"], "--host", status["listen_host"],
               "--port", str(status["port"]), "--served-model-name", status["served_model_name"],
               "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "16384",
               "--gpu-memory-utilization", "0.85", "--max-num-seqs", "1",
               "--limit-mm-per-prompt", '{"image":16,"video":0}',
               "--mm-processor-kwargs", '{"min_pixels":3136,"max_pixels":200704}']
    print(json.dumps({"event": "starting_own_slurm_service", "host": status["listen_host"],
                      "port": status["port"], "model": status["served_model_name"]}), flush=True)
    os.chdir(root)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, importlib.metadata.PackageNotFoundError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(3)
