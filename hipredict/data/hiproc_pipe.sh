#!/usr/bin/env bash
    # Full HiRISE hiproc pipeline: .IMG -> jitter-corrected .slither.cub for specified CCDs
    set -euo pipefail
    shopt -s nullglob  # prevents glob from returning literal string if no match

    LOG_ROOT="${HIPREDICT_LOG_DIR:-${HIPREDICT_FILES_DIR:-files}/hiproc_logs}"
    mkdir -p "$LOG_ROOT"
    LOG_ROOT="$(cd "$LOG_ROOT" && pwd)"

    # ---- Error codes ----
    # 10 - non-fatal data/structure issue (do not add to fail list)
    # 11 - HiCal / calibration failure (add to fail list)
    # 12 - downstream HiPROC failure (stitch / color / jit / slither) (add to fail list)
    # 13 - disk / quota exceeded (system issue, do not add to fail list)
    E_CANT_CD=10
    E_MISSING_IMGS=10
    E_EDR_STATS=10
    E_HICAL=11
    E_HISTITCH=12
    E_HICCDSTITCH=12
    E_HICOLORINIT=12
    E_HIJITREG=12
    E_HISLITHER=12
    E_DISK_FULL=13
    E_QUOTA=13

    HIPROC_VERBOSE_FLAGS=()
    if [[ "${HIPROC_VERBOSE:-0}" != "0" ]]; then
        HIPROC_VERBOSE_FLAGS=(-vvv)
    fi

    fail() {
        local code="$1"
        shift
        echo "HIPROC_FAIL code=${code} msg=$*" >&2
        exit "$code"
    }

    check_disk_error() {
        local logfile="$1"
        local stage="$2"

        if grep -q "No space left on device" "$logfile"; then
            fail "$E_DISK_FULL" "$stage: disk full (ENOSPC)"
        elif grep -q "Disk quota exceeded" "$logfile"; then
            fail "$E_QUOTA" "$stage: quota exceeded (EDQUOT)"
        fi
    }

    # Function: convert raw IMG to ISIS cube and radiometrically calibrate.
    convert_and_calibrate(){
    local img="$1"
    local base=$(basename "${img%.IMG}")
    local cub="${base}_raw.cub"
    local proc="${base}_pr.cub"

    # Logs per step
    local hical_log="${LOG_DIR}/${base}_hical.log"
    local edrstat_log="${LOG_DIR}/${base}_edrstats.log"
    local catlab_log="${LOG_DIR}/${base}_catlab.log"

    if ! EDR_Stats "$img" -o "$cub" >"$edrstat_log" 2>&1; then
        fail "$E_EDR_STATS" "EDR_Stats failed for $img"
    fi

    if ! catlab from="$cub" >"$catlab_log" 2>&1; then
        fail "$E_HICAL" "catlab failed for $cub"
    fi

    if ! HiCal "$cub" --output "$proc" --logfile "$hical_log" "${HIPROC_VERBOSE_FLAGS[@]}" >"$hical_log" 2>&1; then
        check_disk_error "$hical_log" "HiCal"
        fail "$E_HICAL" "HiCal failed for $cub"
    fi

    rm -f "$cub"

    echo "$proc"
    }

    # Function: stitch two calibrated cubes for a given CCD.
    stitch_ccd(){
    local obsid="$1"
    local ccd="$2"
    local proc0="$3"
    local proc1="$4"
    local stitch="${obsid}_${ccd}.stitched.cub"
    local histitch_log="${LOG_DIR}/${obsid}_${ccd}_histitch.log"


    if ! HiStitch "$proc0" "$proc1" -o "$stitch" >"$histitch_log" 2>&1; then
        check_disk_error "$histitch_log" "HiStitch"
        fail "$E_HISTITCH" "HiStitch failed for $proc0 and $proc1"
    fi

    rm -f "$proc0" "$proc1"
    echo "$stitch"
    }

    cleanup_intermediates(){
        local obsid="$1" f
        # Remove known cube intermediates owned by this observation only.
        # Retain raw .IMG, final cubes, metadata and unrelated user files.
        for f in "${obsid}"_*; do
            [[ -f "$f" || -L "$f" ]] || continue
            case "$f" in
                *_raw.cub|*_pr.cub|*.stitched.cub|*.stitched.balance.cub|*.precolor.cub|*.slither.cub)
                    rm -f -- "$f" ;;
            esac
        done
    }

    hiproc_pipeline(){
        local obs="$1"
        local file_loc="$2"
        if ! cd "$file_loc"; then
            fail "$E_CANT_CD" "Cannot cd to $file_loc"
        fi
        file_loc="$PWD"
        
        local obsid
        obsid=$(basename "$file_loc")

        LOG_DIR="${LOG_ROOT}/${obsid}"
        mkdir -p "$LOG_DIR"

        # define your colour groups
        declare -A colour_groups=(
        [RED]="RED3 RED4 RED5"
        [IR]="IR10"
        [BG]="BG12"
        )

        balance_list=()

        # for every colour: convert/calibrate/stitch each CCD, then hiccdstitch that group
        for colour in "${!colour_groups[@]}"; do
            stitch_outputs=()

            # ALL CCDs: convert, calibrate, stitch channels
            for ccd in ${colour_groups[$colour]}; do
                imgs=( "${file_loc}"/*_${ccd}_0.IMG "${file_loc}"/*_${ccd}_1.IMG )
                if [ "${#imgs[@]}" -ne 2 ]; then
                    fail "$E_MISSING_IMGS" "Found ${#imgs[@]} IMG files for $ccd; expected 2."
                fi

                proc0=$(convert_and_calibrate "${imgs[0]}")
                proc1=$(convert_and_calibrate "${imgs[1]}")

                if [ ! -f "$proc0" ]; then
                    fail "$E_HICAL" "Expected proc cube not found: $proc0"
                fi
                if [ ! -f "$proc1" ]; then
                    fail "$E_HICAL" "Expected proc cube not found: $proc1"
                fi

                stitch_cube=$(stitch_ccd "$obsid" "$ccd" "$proc0" "$proc1")
                if [ ! -f "$stitch_cube" ]; then
                    fail "$E_HISTITCH" "Expected stitched cube not found: $stitch_cube"
                fi

                stitch_outputs+=( "$stitch_cube" )
            done

            # RED CCD: hiccdstitch
            if [ "${#stitch_outputs[@]}" -gt 1 ]; then      # If num items in stitch_outputs > 1 (i.e. red), call hiccdstitch
                hiccdstitch_log="${LOG_DIR}/${obsid}_${colour}_hiccdstitch.log"
                if ! HiccdStitch "${stitch_outputs[@]}" > "$hiccdstitch_log" 2>&1; then
                    check_disk_error "$hiccdstitch_log" "HiccdStitch"
                    fail "$E_HICCDSTITCH" "HiccdStitch failed for $colour"
                fi

                # remove the per‐CCD stitched cubes
                rm -f -- "${stitch_outputs[@]}"

                # find the .balance.cub files this produced
                for c in "${stitch_outputs[@]}"; do
                    balance_list+=( "${c%.cub}.balance.cub" )
                done

            # COL CCD: no other stitching
            else                # only one CCD in this colour: treat the single stitched cube as already "balanced"
                local single="${stitch_outputs[0]}"
                local balance="${single%.cub}.balance.cub"
                mv "$single" "$balance"
                balance_list+=( "$balance" )
            fi

        done

        local colour_log="${LOG_DIR}/${obsid}_color.log"
        local hijitreg_log="${LOG_DIR}/${obsid}_hijitreg.log"
        local slither_log="${LOG_DIR}/${obsid}_slither.log"

        # At this stage there is a *.balance.cub for every colour
        if ! HiColorInit ./*.balance.cub "${HIPROC_VERBOSE_FLAGS[@]}" > "$colour_log" 2>&1; then
            check_disk_error "$colour_log" "HiColorInit"
            fail "$E_HICOLORINIT" "HiColorInit failed for ${obsid}"
        fi

        if ! HiJitReg ./*.balance.cub ./*.precolor.cub "${HIPROC_VERBOSE_FLAGS[@]}" > "$hijitreg_log" 2>&1; then
            check_disk_error "$hijitreg_log" "HiJitReg"
            fail "$E_HIJITREG" "HiJitReg failed for ${obsid}"
        fi

        if ! HiSlither ./*.balance.cub ./*.precolor.cub "${HIPROC_VERBOSE_FLAGS[@]}" > "$slither_log" 2>&1; then
            check_disk_error "$slither_log" "HiSlither"
            fail "$E_HISLITHER" "HiSlither failed for ${obsid}"
        fi

        # RENAMING FINAL CUBEFILES
        # 1) RED balance cubes (they still have “.stitched.balance.cub” in their names)
        for f in *RED*.stitched.balance.cub; do
            # strip prefix and suffix to get just the CCD
            ccd="${f#${obsid}_}"            # e.g. RED3.stitched.balance.cub
            ccd="${ccd%.stitched.balance.cub}"  # e.g. RED3
            new="${obsid}_${ccd}_processed.cub"
            mv "$f" "$new"
        done

        # 2) Colour slither cubes (only BG & IR)
        for f in *.slither.cub; do
            local ccd="${f#${obsid}_}"            # e.g. BG12.slither.cub
            ccd="${ccd%.slither.cub}"       # e.g. BG12
            local new="${obsid}_${ccd}_processed.cub"
            mv "$f" "$new"
        done

        # CLEANUP
        if [[ "${HIPROC_VERBOSE:-0}" != "0" ]]; then
        echo -e "\n\n"
        echo "Pipeline completed. Final processed cubes:"
        ls *_processed.cub
        fi

        cleanup_intermediates "$obsid"
    }

    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
        if [[ $# -ne 2 ]]; then
            echo "Usage: $0 <observation_id> <observation_dir>" >&2
            exit 64
        fi

        hiproc_pipeline "$1" "$2"
    fi
