"""Confine model SDK caches and temporary files to this own project process."""
import os
from pathlib import Path
import tempfile


def configure_model_environment(project):
    project=Path(project).resolve()
    base=project/".runtime/model_cache"
    folders={"HF_HOME":"huggingface","HF_HUB_CACHE":"huggingface/hub",
             "HUGGINGFACE_HUB_CACHE":"huggingface/hub","HF_ASSETS_CACHE":"huggingface/assets",
             "TORCH_HOME":"torch","TORCH_EXTENSIONS_DIR":"torch_extensions",
             "XDG_CACHE_HOME":"xdg","TMPDIR":"tmp","TEMP":"tmp","TMP":"tmp"}
    for name,relative in folders.items():
        path=(base/relative).resolve()
        if not path.is_relative_to(project):raise RuntimeError("Model cache symlink escapes the own project: "+name)
        path.mkdir(parents=True,exist_ok=True)
        os.environ[name]=str(path)
    # Avoid a pre-existing legacy variable redirecting Transformers elsewhere.
    os.environ.pop("TRANSFORMERS_CACHE",None)
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    tempfile.tempdir=os.environ["TMPDIR"]
    return {name:os.environ[name] for name in folders}
