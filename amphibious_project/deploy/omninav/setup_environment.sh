#!/usr/bin/env bash
set -euo pipefail
exec "${OMNINAV_BASE_PYTHON:-python3}" "$(dirname -- "${BASH_SOURCE[0]}")/setup_environment.py" "$@"
