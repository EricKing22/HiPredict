#!/usr/bin/env bash
# Source this script to activate the existing, patched HiPROC + ISIS backend.
_hipredict_activate_backend() {
    local conda_sh="${HIPREDICT_CONDA_SH:-}" candidate root status had_nounset=0
    if [[ -z "$conda_sh" && -n "${CONDA_EXE:-}" ]]; then
        root="$(dirname "$(dirname "$CONDA_EXE")")"
        [[ ! -f "$root/etc/profile.d/conda.sh" ]] || conda_sh="$root/etc/profile.d/conda.sh"
    fi
    if [[ -z "$conda_sh" ]]; then
        for candidate in "$HOME/miniforge3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh"; do
            if [[ -f "$candidate" ]]; then conda_sh="$candidate"; break; fi
        done
    fi
    if [[ ! -f "$conda_sh" ]]; then
        echo "ERROR: Set HIPREDICT_CONDA_SH to your conda.sh activation script." >&2
        return 1
    fi
    export HIPREDICT_CONDA_SH="$conda_sh"
    export HIPREDICT_ISIS_ENV="${HIPREDICT_ISIS_ENV:-isis}"
    export HIPREDICT_HIPROC_ENV="${HIPREDICT_HIPROC_ENV:-hiproc}"
    [[ $- != *u* ]] || had_nounset=1
    set +u
    if source "$conda_sh"; then
        conda deactivate >/dev/null 2>&1 || true
        if conda activate "$HIPREDICT_ISIS_ENV" && conda activate --stack "$HIPREDICT_HIPROC_ENV"; then
            status=0
        else
            status=1
        fi
    else
        status=1
    fi
    (( had_nounset == 0 )) || set -u
    if (( status != 0 )); then echo "ERROR: Could not activate the HiPROC + ISIS environments." >&2; fi
    return "$status"
}
_hipredict_activate_backend || { return 1 2>/dev/null || exit 1; }
