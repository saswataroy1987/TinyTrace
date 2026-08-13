#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_phase_a_local.sh --dataset-root /path/to/TinyTrace_kaggle_dataset_ready [options]

Options:
  --dataset-root PATH       Required. Local copy of TinyTrace_kaggle_dataset_ready.
  --work-root PATH          Writable run directory. Default: ./phase_a_v5_local_run
  --venv PATH               Python virtualenv directory. Default: ./.venv-phase-a
  --python BIN              Python executable used to create the venv. Default: python3
  --device DEVICE           Training device. Default: cuda
  --persist-output-root     Directory to copy final artifacts into. Default: <work-root>/exported_artifacts
  --skip-feature-cache      Reuse an existing feature cache in the same work root.
  --skip-install            Skip dependency installation and reuse the existing venv.
  --force-recreate-venv     Delete and recreate the venv before installing.
  --help                    Show this message.

Examples:
  ./TinyTrace/scripts/run_phase_a_local.sh \
    --dataset-root ../TinyTrace_kaggle_dataset_ready

  ./TinyTrace/scripts/run_phase_a_local.sh \
    --dataset-root ../TinyTrace_kaggle_dataset_ready \
    --work-root ./phase_a_v5_local_run \
    --skip-feature-cache
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

DATASET_ROOT=""
WORK_ROOT="${REPO_ROOT}/phase_a_v5_local_run"
VENV_DIR="${REPO_ROOT}/.venv-phase-a"
PYTHON_BIN="python3"
DEVICE="cuda"
PERSIST_OUTPUT_ROOT=""
SKIP_FEATURE_CACHE=0
SKIP_INSTALL=0
FORCE_RECREATE_VENV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --work-root)
      WORK_ROOT="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --persist-output-root)
      PERSIST_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --skip-feature-cache)
      SKIP_FEATURE_CACHE=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --force-recreate-venv)
      FORCE_RECREATE_VENV=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "--dataset-root is required." >&2
  usage >&2
  exit 1
fi

DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi
mkdir -p "${WORK_ROOT}"
WORK_ROOT="$(cd "${WORK_ROOT}" && pwd)"

if [[ -z "${PERSIST_OUTPUT_ROOT}" ]]; then
  PERSIST_OUTPUT_ROOT="${WORK_ROOT}/exported_artifacts"
fi
mkdir -p "${PERSIST_OUTPUT_ROOT}"
PERSIST_OUTPUT_ROOT="$(cd "${PERSIST_OUTPUT_ROOT}" && pwd)"

if [[ ${FORCE_RECREATE_VENV} -eq 1 && -d "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Virtualenv python not found at ${VENV_DIR}/bin/python" >&2
  exit 1
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

if [[ ${SKIP_INSTALL} -eq 0 ]]; then
  "${VENV_PIP}" install --upgrade pip
  if [[ "${DEVICE}" == cuda* ]]; then
    "${VENV_PIP}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  else
    "${VENV_PIP}" install torch torchvision torchaudio
  fi
  "${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/setup_kaggle_env.py" --dataset-root "${DATASET_ROOT}"
fi

echo
echo "Runtime configuration:"
echo " dataset_root=${DATASET_ROOT}"
echo " work_root=${WORK_ROOT}"
echo " persist_output_root=${PERSIST_OUTPUT_ROOT}"
echo " venv=${VENV_DIR}"
echo " skip_feature_cache=${SKIP_FEATURE_CACHE}"
echo " skip_install=${SKIP_INSTALL}"
echo " force_recreate_venv=${FORCE_RECREATE_VENV}"
echo
echo "Python environment:"
"${VENV_PYTHON}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
echo

COMMAND=(
  "${VENV_PYTHON}"
  "${PROJECT_ROOT}/scripts/run_phase_a_kaggle.py"
  --dataset-root "${DATASET_ROOT}"
  --work-root "${WORK_ROOT}"
  --device "${DEVICE}"
  --persist-output-root "${PERSIST_OUTPUT_ROOT}"
)

if [[ ${SKIP_FEATURE_CACHE} -eq 1 ]]; then
  COMMAND+=(--skip-feature-cache)
fi

echo "Starting TinyTrace Phase A local run:"
printf ' %q' "${COMMAND[@]}"
echo
echo

PYTHONPATH="${PROJECT_ROOT}" "${COMMAND[@]}"
