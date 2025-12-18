#!/bin/bash
# Run RD-Agent locally without GPU request
# Supports GPT-5 and 5-day configurations

set -e

# Display usage information
usage() {
    cat << EOF
Usage: $0 [OPTION]

Run RD-Agent with specified configuration.

Options:
  gpt5      Run with GPT-5 (default)
  3day      Run with 5-day time limit on GPT-5
  help      Display this help message

Examples:
  $0 gpt5       # Run RD-Agent with GPT-5
  $0 3day       # Run RD-Agent with 5-day configuration on GPT-5
EOF
    exit 0
}

# Default configuration
AGENT_VARIANT="gpt5"
CHAT_MODEL="azure/gpt-5"
RUNS_DIR_SUFFIX=""
COMPETITION_SET="experiments/splits/usenhance.txt"
CONFIG_FILE=""

# Parse command line arguments
case "${1:-gpt5}" in
    gpt5)
        AGENT_VARIANT="gpt5"
        CHAT_MODEL="azure/gpt-5"
        ;;
    3day)
        AGENT_VARIANT="gpt5"
        CHAT_MODEL="azure/gpt-5"
        RUNS_DIR_SUFFIX="/3_day"
        CONFIG_FILE="config_3_days.yaml"
        ;;
    help)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use 'help' for usage information"
        exit 1
        ;;
esac

# Change to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"
cd "$PROJECT_DIR"

echo "=============================================="
echo "Running RD-Agent"
echo "=============================================="
echo "Agent Variant: ${AGENT_VARIANT}"
echo "Chat Model: ${CHAT_MODEL}"
echo "Started at: $(date)"
echo ""

# Load environment variables from the centralized .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from centralized .env"
    set -a
    source .env
    set +a
else
    echo "Warning: .env not found in project root"
fi

# Set backend and model for RD-Agent
export BACKEND="rdagent.oai.backend.litellm.LiteLLMAPIBackend"
export CHAT_MODEL="${CHAT_MODEL}"
echo "Using backend: ${BACKEND}"
echo "Using model: ${CHAT_MODEL}"
echo ""

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate rexagent

# Set runs directory if 5-day variant
if [ -n "$RUNS_DIR_SUFFIX" ]; then
    export RUNS_DIR="${RUNS_DIR:-.}/runs/${RUNS_DIR_SUFFIX}"
else
    export RUNS_DIR="${RUNS_DIR:-.}/runs/"
fi

# Build Python command
PYTHON_CMD="python -u run_agent_rdagent.py \
    --agent-id rdagent \
    --competition-set ${COMPETITION_SET} \
    --n-workers 1"

# Add config file if specified
if [ -n "$CONFIG_FILE" ]; then
    PYTHON_CMD="$PYTHON_CMD \
    --config-file $CONFIG_FILE"
fi

# Run the Python script with rdagent-specific runner
echo "Executing: $PYTHON_CMD"
echo ""
eval "$PYTHON_CMD"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed at $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
