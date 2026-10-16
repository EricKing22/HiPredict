#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE_SCRIPT="$SCRIPT_DIR/activate_hiproc_backend.sh"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

print_context() {
    echo "hiproc+isis backend check"
    echo "  HIPREDICT_CONDA_SH=${HIPREDICT_CONDA_SH:-<unset>}"
    echo "  HIPREDICT_ISIS_ENV=${HIPREDICT_ISIS_ENV:-<unset>}"
    echo "  HIPREDICT_HIPROC_ENV=${HIPREDICT_HIPROC_ENV:-<unset>}"
    echo "  ISISROOT=${ISISROOT:-<unset>}"
    echo "  ISISDATA=${ISISDATA:-<unset>}"
}

verify_backend() {
    if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
        fail "Activation script not found at '$ACTIVATE_SCRIPT'."
    fi

    if ! source "$ACTIVATE_SCRIPT"; then
        echo "Backend activation failed." >&2
        echo "Check the README preprocessing instructions and confirm the conda path and environment names." >&2
        fail "Could not activate the stacked hiproc+isis backend."
    fi

    required_commands=(python EDR_Stats catlab HiCal HiStitch HiccdStitch HiColorInit HiJitReg HiSlither handmos editlab getkey spiceinit)
    missing_commands=()

    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_commands+=("$cmd")
        fi
    done

    if (( ${#missing_commands[@]} > 0 )); then
        echo "Backend activation succeeded, but required commands are missing:" >&2
        printf '  %s\n' "${missing_commands[@]}" >&2
        echo >&2

        if printf '%s\n' "${missing_commands[@]}" | grep -Eq 'EDR_Stats|catlab|handmos|editlab|getkey|spiceinit'; then
            echo "Likely issue: the ISIS environment is incomplete or not activating correctly." >&2
        fi
        if printf '%s\n' "${missing_commands[@]}" | grep -Eq 'HiCal|HiStitch|HiccdStitch|HiColorInit|HiJitReg|HiSlither'; then
            echo "Likely issue: the hiproc environment is incomplete, hiproc was not installed, or the stacked activation failed." >&2
        fi
        echo "Check the README preprocessing instructions, then re-run this verifier." >&2
        fail "hiproc+isis backend is not ready."
    fi

    if [[ -z "${ISISROOT:-}" || -z "${ISISDATA:-}" ]]; then
        echo "Backend commands are available, but ISISROOT or ISISDATA is unset." >&2
        echo "The backend may be only partially configured. Check the README preprocessing instructions." >&2
        fail "ISIS environment variables are not fully configured."
    fi

    hiproc_version="$(python -c 'from importlib.metadata import version; print(version("hiproc"))' 2>/dev/null || echo unknown)"
    if [[ "$hiproc_version" != "0.11.0" ]]; then
        echo "WARNING: found hiproc version $hiproc_version, but HiPredict's training data were processed with hiproc 0.11.0." >&2
        echo "Cubes from other versions may differ; see docs/preprocessing.md." >&2
    fi

    isis_version="unknown"
    for meta in "$ISISROOT"/conda-meta/isis-[0-9]*.json; do
        [[ -e "$meta" ]] && isis_version="$(basename "$meta" .json | cut -d- -f2)"
    done
    if [[ "$isis_version" != "9.0.0" ]]; then
        echo "WARNING: found ISIS version $isis_version, but HiPredict's training data were processed with ISIS 9.0.0." >&2
        echo "Cubes from other versions may differ; see docs/preprocessing.md." >&2
    fi

    echo "hiproc+isis commands and environment verified (hiproc $hiproc_version, ISIS $isis_version)"
    print_context
}

verify_backend
