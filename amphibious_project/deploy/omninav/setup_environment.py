"""Install only the own L17 model environment; --dry-run prints a reviewable plan."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

PROJECT=Path(__file__).resolve().parents[2]


def choose_python(explicit=None):
    candidates=[explicit] if explicit else [sys.executable,"python3.10","python3.11","python3.12","python3",
                                           "/home/ubuntu/qsq/isaacsim/kit/python/bin/python3"]
    for candidate in candidates:
        if not candidate:continue
        executable=shutil.which(candidate)
        if not executable:continue
        result=subprocess.run([executable,"-B","-c","import sys,json,ensurepip; print(json.dumps(list(sys.version_info[:2])))"],capture_output=True,text=True)
        if result.returncode==0 and [3,10]<=json.loads(result.stdout)<[3,13]:return executable
    raise RuntimeError("A Python 3.10, 3.11 or 3.12 interpreter with ensurepip is required; --base-python can select an existing compatible interpreter")


def install_plan(project,base_python,torch_index_url="https://download.pytorch.org/whl/cu124"):
    project=Path(project).resolve()
    index=urlsplit(torch_index_url)
    if index.scheme!="https" or not index.hostname or index.username or index.password:
        raise ValueError("Use an HTTPS package index without embedded credentials")
    paths={name:(project/relative).resolve() for name,relative in {
        "environment":".venvs/omninav","pip_cache":".runtime/model_cache/pip",
        "hf_cache":".runtime/model_cache/huggingface","torch_cache":".runtime/model_cache/torch",
        "torch_extensions":".runtime/model_cache/torch_extensions","temporary":".runtime/model_cache/tmp"}.items()}
    if any(not path.is_relative_to(project) for path in paths.values()):raise RuntimeError("An installation/cache symlink escapes the own project")
    fork=(project/"third_party/OmniNav/train_code/transformers-main").resolve()
    if not fork.is_relative_to(project):raise RuntimeError("Editable Transformers source must remain inside the own project")
    python=str(paths["environment"]/"bin/python")
    pip=[python,"-m","pip","install","--disable-pip-version-check"]
    commands=[[base_python,"-m","venv",str(paths["environment"])],
              pip+["pip==25.0.1","setuptools==75.8.0","wheel==0.45.1"],
              pip+["torch==2.6.0","torchvision==0.21.0","--index-url",torch_index_url],
              pip+["-r",str(project/"deploy/omninav/requirements-inference.txt")],
              pip+["--no-build-isolation","-e",str(fork)],
              [python,"-m","pip","check"],
              [python,"-B",str(project/"deploy/omninav/check_environment.py"),"--output",str(project/"results/deployment/environment.json")]]
    return paths,commands


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--preflight",action="store_true",help="Verify the unpacked archive and transferred model before installation")
    parser.add_argument("--base-python",default=os.environ.get("OMNINAV_BASE_PYTHON"))
    parser.add_argument("--torch-index-url",default=os.environ.get("OMNINAV_TORCH_INDEX_URL","https://download.pytorch.org/whl/cu124"),
                        help="HTTPS mirror for the pinned Torch/TorchVision wheels; default is the official CUDA 12.4 index")
    args=parser.parse_args()
    base=choose_python(args.base_python)
    paths,commands=install_plan(PROJECT,base,args.torch_index_url)
    if args.preflight:commands.insert(0,[base,"-B",str(PROJECT/"deploy/verify_deployment.py"),"--require-manifest"])
    if args.dry_run:
        print(json.dumps({"project":str(PROJECT),"target_platform":"Linux","base_python":base,
                          "commands":commands,"gpu_process_started":False},indent=2));return
    allowed=[Path("/home/ubuntu/sdc/L17"),Path("/home/wangjiong/L17")]
    if os.name!="posix" or not any(PROJECT.is_relative_to(root.resolve()) for root in allowed if root.exists()):
        raise RuntimeError("Run the installation only inside the own L17 on the Linux server")
    for key,path in paths.items():
        if key!="environment":path.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(PYTHONNOUSERSITE="1",PIP_REQUIRE_VIRTUALENV="true",PIP_CACHE_DIR=str(paths["pip_cache"]),
               PIP_CONFIG_FILE=os.devnull,PIP_EXTRA_INDEX_URL="",
               HF_HOME=str(paths["hf_cache"]),TORCH_HOME=str(paths["torch_cache"]),
               TORCH_EXTENSIONS_DIR=str(paths["torch_extensions"]),TMPDIR=str(paths["temporary"]))
    logs=PROJECT/"results/deployment";logs.mkdir(parents=True,exist_ok=True)
    for index,command in enumerate(commands):
        print("Install step",index+1,"/",len(commands),flush=True)
        with (logs/f"install_{index+1:02d}.log").open("w") as handle:
            result=subprocess.run(command,cwd=PROJECT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError("Installation failed; inspect "+str(logs/f"install_{index+1:02d}.log"))
    print("Own environment and CPU preflight passed. No simulator, model inference or GPU job was started.")


if __name__=="__main__":main()
