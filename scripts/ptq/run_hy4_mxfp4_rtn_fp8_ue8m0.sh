#!/usr/bin/env bash
# HY4 BF16 -> data-free MXFP4-RTN routed experts -> FP8 UE8M0 Linear/MTP.
# Process management (setsid/PID) belongs in a per-run launcher.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/Hy4/ptq/mxfp4_rtn_fp8_ue8m0/hy4_mxfp4_rtn.yaml}"
BF16_MODEL_PATH="${BF16_MODEL_PATH:-}"
STAGE1_MODEL="${STAGE1_MODEL:-}"
FINAL_MODEL="${FINAL_MODEL:-}"
LOG_DIR="${LOG_DIR:-${FINAL_MODEL}/logs}"
NUM_WORKERS="${NUM_WORKERS:-8}"

do_rtn=1
do_fp8=1
do_sync=1
do_validate=1
use_cpu=0
full_duplicate_check=0

usage() {
    cat <<EOF
Usage: bash $0 [options]

Required environment:
  BF16_MODEL_PATH   Source BF16 model
  STAGE1_MODEL      Stage-1 MXFP4-RTN model directory
  FINAL_MODEL       Final model directory

Optional environment:
  CONFIG            Stage-1 RTN YAML
  LOG_DIR           Log directory (default: FINAL_MODEL/logs)
  NUM_WORKERS       Conversion workers (default: 8)

Options:
  --skip-rtn
  --skip-fp8
  --skip-config-sync
  --skip-validate
  --validate-only
  --cpu
  --full-duplicate-check
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --skip-rtn) do_rtn=0 ;;
        --skip-fp8) do_fp8=0 ;;
        --skip-config-sync) do_sync=0 ;;
        --skip-validate) do_validate=0 ;;
        --validate-only)
            do_rtn=0
            do_fp8=0
            do_sync=0
            do_validate=1
            ;;
        --cpu) use_cpu=1 ;;
        --full-duplicate-check) full_duplicate_check=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

[[ -n "${BF16_MODEL_PATH}" ]] || { echo "BF16_MODEL_PATH is required" >&2; exit 2; }
[[ -n "${STAGE1_MODEL}" ]] || { echo "STAGE1_MODEL is required" >&2; exit 2; }
[[ -n "${FINAL_MODEL}" ]] || { echo "FINAL_MODEL is required" >&2; exit 2; }

for path in "${REPO_ROOT}" "${CONFIG}" "${BF16_MODEL_PATH}"; do
    [[ -e "${path}" ]] || { echo "Required path does not exist: ${path}" >&2; exit 1; }
done

if [[ "${do_rtn}" -eq 1 ]] \
    && [[ -d "${STAGE1_MODEL}" ]] \
    && find "${STAGE1_MODEL}" -mindepth 1 -print -quit | grep -q .; then
    echo "Stage-1 output is not empty: ${STAGE1_MODEL}" >&2
    exit 1
fi
if [[ "${do_rtn}" -eq 0 && ! -f "${STAGE1_MODEL}/model.safetensors.index.json" ]]; then
    echo "Missing Stage-1 checkpoint: ${STAGE1_MODEL}" >&2
    exit 1
fi
if [[ "${do_fp8}" -eq 1 && -f "${FINAL_MODEL}/model.safetensors.index.json" ]]; then
    echo "Final model already exists: ${FINAL_MODEL}" >&2
    exit 1
fi
if [[ "${do_fp8}" -eq 0 && "${do_validate}" -eq 1 \
    && ! -f "${FINAL_MODEL}/model.safetensors.index.json" ]]; then
    echo "Missing final checkpoint: ${FINAL_MODEL}" >&2
    exit 1
fi

mkdir -p "${FINAL_MODEL}" "${LOG_DIR}" "$(dirname "${STAGE1_MODEL}")"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

if [[ "${do_rtn}" -eq 1 ]]; then
    echo "[1/4] Data-free MXFP4 RTN routed experts"
    stage1_args=(
        python3 tools/hy4_mxfp4_weight_only.py
        -c "${CONFIG}"
        --input-path "${BF16_MODEL_PATH}"
        --output-path "${STAGE1_MODEL}"
        --num-workers "${NUM_WORKERS}"
    )
    [[ "${use_cpu}" -eq 0 ]] || stage1_args+=(--cpu)
    "${stage1_args[@]}" 2>&1 | tee "${LOG_DIR}/stage1_mxfp4_rtn.log"
else
    echo "[1/4] Skip RTN"
fi

if [[ "${do_fp8}" -eq 1 ]]; then
    [[ -f "${STAGE1_MODEL}/model.safetensors.index.json" ]] || {
        echo "Stage 1 did not produce a checkpoint: ${STAGE1_MODEL}" >&2
        exit 1
    }
    build_dir="${FINAL_MODEL}/.model_build"
    if [[ -d "${build_dir}" ]] && find "${build_dir}" -mindepth 1 -print -quit | grep -q .; then
        echo "Temporary build directory is not empty: ${build_dir}" >&2
        exit 1
    fi
    mkdir -p "${build_dir}"
    fp8_args=(
        python3 tools/hy4_mxfp4_rtn_to_fp8_ue8m0.py
        --input-path "${STAGE1_MODEL}"
        --output-path "${build_dir}"
        --num-workers "${NUM_WORKERS}"
    )
    [[ "${use_cpu}" -eq 0 ]] || fp8_args+=(--cpu)

    echo "[2/4] FP8 E4M3 + UE8M0 Linear/MTP"
    "${fp8_args[@]}" 2>&1 | tee "${LOG_DIR}/stage2_fp8_ue8m0.log"

    if [[ "${do_sync}" -eq 1 ]]; then
        echo "[3/4] Synchronize BF16 config and auxiliary files"
        python3 tools/sync_hy4_rtn_config_to_fp8.py \
            "${build_dir}" \
            --reference "${BF16_MODEL_PATH}" \
            --aux-source "${BF16_MODEL_PATH}" \
            2>&1 | tee "${LOG_DIR}/stage3_sync_config.log"
    else
        echo "[3/4] Skip config sync"
    fi

    find "${build_dir}" -mindepth 1 -maxdepth 1 -exec mv -t "${FINAL_MODEL}" -- {} +
    rmdir "${build_dir}"
elif [[ "${do_sync}" -eq 1 ]]; then
    echo "[3/4] Synchronize BF16 config and auxiliary files"
    python3 tools/sync_hy4_rtn_config_to_fp8.py \
        "${FINAL_MODEL}" \
        --reference "${BF16_MODEL_PATH}" \
        --aux-source "${BF16_MODEL_PATH}" \
        2>&1 | tee "${LOG_DIR}/stage3_sync_config.log"
else
    echo "[2/4] Skip FP8 conversion"
    echo "[3/4] Skip config sync"
fi

if [[ "${do_validate}" -eq 1 ]]; then
    validate_args=(
        python3 tools/validate_hy4_mxfp4_rtn.py
        --stage1-path "${STAGE1_MODEL}"
        --output-path "${FINAL_MODEL}"
        --bf16-path "${BF16_MODEL_PATH}"
    )
    [[ "${full_duplicate_check}" -eq 0 ]] || validate_args+=(--full-duplicate-check)
    echo "[4/4] Validate final checkpoint"
    "${validate_args[@]}" 2>&1 | tee "${LOG_DIR}/stage4_validate.log"
else
    echo "[4/4] Skip validation"
fi

echo "Done."
echo "Stage-1: ${STAGE1_MODEL}"
echo "Final:   ${FINAL_MODEL}"
