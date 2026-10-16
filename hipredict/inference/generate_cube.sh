#!/bin/bash

set -euo pipefail       # Failures propagate to python via nonzero exit

# ================ ERROR HANDLING ==============================
usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") <input_cube> <neighbor_cube> <output_cube> <ccd_id>
EOF
    exit 2
}

if [[ $# -ne 4 ]]; then
    usage
fi

# ================ INPUTS ==============================
CUB_IN="$1"
NEIGHBOR="$2"
OUT_CUBE="$3"
CCD="$4"

# ================ SANITY CHECKS  ==============================
if [[ ! -f "$CUB_IN" ]]; then
    echo "ERROR: input_cube not found: $CUB_IN" >&2
    exit 1
fi
if [[ ! -f "$NEIGHBOR" ]]; then
    echo "ERROR: neighbor_cube not found: $NEIGHBOR" >&2
    exit 1
fi

OUT_DIR="$(dirname "$OUT_CUBE")"
mkdir -p "$OUT_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE_SCRIPT="$SCRIPT_DIR/../data/activate_hiproc_backend.sh"

# ================ ENVIRONMENT  ==============================
if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    echo "ERROR: backend activation script not found: $ACTIVATE_SCRIPT" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

# ================ PIPELINE ==============================
# Clone the neighbor to keep all BLOBs/Tables/Labels
cp "$NEIGHBOR" "$OUT_CUBE"

# Inject pixels into the clone
handmos from="$CUB_IN" mosaic="$OUT_CUBE" \
        insamp=1 inline=1 outsamp=1 outline=1 \
        create=no matchbandbin=false

# Update the Labels so ISIS knows what this file actually is
editlab from="$OUT_CUBE" options=setkey grpname=Instrument keyword=CcdId value="$CCD"

# Fix the Core Dimensions
SAMPLES=$(getkey from="$CUB_IN" grpname=Dimensions keyword=Samples)
LINES=$(getkey from="$CUB_IN" grpname=Dimensions keyword=Lines)

editlab from="$OUT_CUBE" options=setkey grpname=Dimensions keyword=Samples value=$SAMPLES
editlab from="$OUT_CUBE" options=setkey grpname=Dimensions keyword=Lines value=$LINES

# Re-initialize SPICE (otherwise it will still use the neighbor's pointing)
# WARNING: due to cropping and sub-pixel effects, SPICE kernels unlikely to be perfectly aligned
spiceinit from="$OUT_CUBE"

echo "$OUT_CUBE"
