#!/bin/bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${L17_CONDA_PREFIX:-$HOME/miniconda3/envs/isaacsim4.5}/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "请设置 L17_CONDA_PREFIX，指向已安装 Isaac Sim 4.5 的环境。" >&2
    exit 1
fi
exec "$python_bin" -B "$project_dir/scripts/launch.py" straight "$@"
