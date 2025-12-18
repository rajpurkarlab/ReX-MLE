#!/bin/bash
set -euo pipefail
set -x

cd "${AGENT_DIR}"

WORKSPACE_DIR="${WORKSPACE_DIR:-${PWD}}"
SUBMISSION_DIR="${SUBMISSION_DIR:-${WORKSPACE_DIR}/submission}"
LOGS_DIR="${LOGS_DIR:-${WORKSPACE_DIR}/logs}"
CODE_DIR="${CODE_DIR:-${WORKSPACE_DIR}/code}"

if ! command -v conda &> /dev/null; then
  echo "conda is required to run ML-Master."
  exit 1
fi

eval "$(conda shell.bash hook)"
set +u
conda activate rexagent
set -u

EXP_ID="${COMPETITION_ID:-}"
if [[ -z "${EXP_ID}" ]]; then
  echo "COMPETITION_ID environment variable is required."
  exit 1
fi

DATA_DIR_PATH="${DATA_DIR:-${WORKSPACE_DIR}/data}"
if [[ ! -d "${DATA_DIR_PATH}" ]]; then
  echo "DATA_DIR (${DATA_DIR_PATH}) does not exist."
  exit 1
fi

TIME_LIMIT_SECS="${TIME_LIMIT_SECS:-43200}"
# Allow overrides via env to avoid duplicate flags
TIMEOUT_OVERRIDE="${ML_MASTER_EXEC_TIMEOUT:-}"
STEPS_OVERRIDE="${ML_MASTER_AGENT_STEPS:-}"
AGENT_TIME_LIMIT_OVERRIDE="${ML_MASTER_AGENT_TIME_LIMIT:-}"
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT="$(format_time "${TIME_LIMIT_SECS}")"

GPU_INDEX="${ML_MASTER_GPU_INDEX:-${MEMORY_INDEX:-0}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$(nproc || echo 8)}"
START_CPU_ID="${START_CPU_ID:-0}"

RUN_SUFFIX="$(date +%Y%m%d%H%M%S)"
RUN_NAME="${EXP_ID}_${RUN_SUFFIX}"
WORKSPACES_ROOT="${AGENT_DIR}/workspaces/run"
LOGS_ROOT="${AGENT_DIR}/logs/run"
RUN_WORKSPACE="${WORKSPACES_ROOT}/${RUN_NAME}"
RUN_LOG_DIR="${LOGS_ROOT}/${RUN_NAME}"

GRADING_PORT_FILE="${WORKSPACE_DIR}/.grading_port"
if [[ -f "${GRADING_PORT_FILE}" ]]; then
  GRADING_PORT="$(cat "${GRADING_PORT_FILE}")"
  export ML_MASTER_VALIDATE_URL="http://127.0.0.1:${GRADING_PORT}"
fi

mkdir -p "${WORKSPACES_ROOT}" "${LOGS_ROOT}" "${SUBMISSION_DIR}" "${CODE_DIR}" "${LOGS_DIR}"

FULL_INSTRUCTIONS_DIR="${AGENT_DIR}/dataset/full_instructions/${EXP_ID}"
DESC_FILE="${FULL_INSTRUCTIONS_DIR}/full_instructions.txt"
if [[ ! -f "${DESC_FILE}" ]]; then
  # fallback to provided instructions if custom file missing
  DESC_FILE="${INSTRUCTIONS_FILE}"
fi

declare -a EXTRA_ARGS

maybe_add_arg() {
  local key="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    EXTRA_ARGS+=("${key}=${value}")
  fi
}

maybe_add_arg "agent.code.model" "${ML_MASTER_CODE_MODEL:-}"
maybe_add_arg "agent.code.temp" "${ML_MASTER_CODE_TEMP:-}"
maybe_add_arg "agent.code.base_url" "${ML_MASTER_CODE_BASE_URL:-}"
maybe_add_arg "agent.code.api_key" "${ML_MASTER_CODE_API_KEY:-}"
maybe_add_arg "agent.feedback.model" "${ML_MASTER_FEEDBACK_MODEL:-}"
maybe_add_arg "agent.feedback.temp" "${ML_MASTER_FEEDBACK_TEMP:-}"
maybe_add_arg "agent.feedback.base_url" "${ML_MASTER_FEEDBACK_BASE_URL:-}"
maybe_add_arg "agent.feedback.api_key" "${ML_MASTER_FEEDBACK_API_KEY:-}"
maybe_add_arg "agent.steps" "${STEPS_OVERRIDE}"
maybe_add_arg "agent.time_limit" "${AGENT_TIME_LIMIT_OVERRIDE}"
maybe_add_arg "exec.timeout" "${TIMEOUT_OVERRIDE}"

ML_MASTER_CMD=(
  python main_mcts.py
  dataset_dir="${DATA_DIR_PATH}"
  data_dir="${DATA_DIR_PATH}"
  desc_file="${DESC_FILE}"
  exp_name="${RUN_NAME}"
  start_cpu_id="${START_CPU_ID}"
  cpu_number="${CPUS_PER_TASK}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

set +e
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" timeout "${TIME_LIMIT_SECS}" "${ML_MASTER_CMD[@]}"
RUN_EXIT=$?
set -e

if [[ ${RUN_EXIT} -eq 124 ]]; then
  echo "ML-Master timed out after ${TIME_LIMIT}"
fi

copy_if_exists() {
  local src="$1"
  local dest="$2"
  if [[ -f "${src}" ]]; then
    cp "${src}" "${dest}"
    return 0
  fi
  return 1
}

RUN_BEST_SUBMISSION="${RUN_WORKSPACE}/best_submission"
if [[ -d "${RUN_BEST_SUBMISSION}" ]]; then
  if [[ -f "${RUN_BEST_SUBMISSION}/submission.csv" ]]; then
    rm -f "${SUBMISSION_DIR}/submission.csv"
    cp "${RUN_BEST_SUBMISSION}/submission.csv" "${SUBMISSION_DIR}/submission.csv"
    if compgen -G "${RUN_BEST_SUBMISSION}/*" > /dev/null; then
      if command -v rsync &> /dev/null; then
        rsync -a --exclude 'submission.csv' "${RUN_BEST_SUBMISSION}/" "${SUBMISSION_DIR}/"
      else
        cp -r "${RUN_BEST_SUBMISSION}/." "${SUBMISSION_DIR}/"
      fi
    fi
    echo "Copied best submission to ${SUBMISSION_DIR}"
  else
    echo "No submission.csv found in ${RUN_BEST_SUBMISSION}"
  fi
else
  echo "Best submission directory not found at ${RUN_BEST_SUBMISSION}"
fi

RUN_BEST_SOLUTION="${RUN_LOG_DIR}/best_solution.py"
if copy_if_exists "${RUN_BEST_SOLUTION}" "${CODE_DIR}/best_solution.py"; then
  echo "Copied best_solution.py to ${CODE_DIR}"
else
  echo "best_solution.py not found in ${RUN_LOG_DIR}"
fi

if [[ -d "${RUN_LOG_DIR}" ]]; then
  DEST_LOG_DIR="${LOGS_DIR}/$(basename "${RUN_LOG_DIR}")"
  rm -rf "${DEST_LOG_DIR}"
  cp -r "${RUN_LOG_DIR}" "${DEST_LOG_DIR}"
  echo "Copied logs to ${DEST_LOG_DIR}"
fi

exit "${RUN_EXIT}"
