#!/usr/bin/env bash
# Source the activation script distributed with the Python package.
_hipredict_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_hipredict_script_dir/../hipredict/data/activate_hiproc_backend.sh" || { return 1 2>/dev/null || exit 1; }
