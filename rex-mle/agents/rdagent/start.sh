#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                     RD-Agent Runner for Med-MLE-Bench                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting at: $(date)"
echo "Competition: ${COMPETITION_ID}"
echo "Working directory: $(pwd)"
echo ""

# ==============================================================================
# Configuration
# ==============================================================================
RDAGENT_DIR="${AGENT_DIR}"
RDAGENT_ENV_FILE="${AGENT_DIR}/.env"

# Parse arguments passed from run_agent_no_docker.py
# Arguments can come in two formats:
# 1. --key value (argparse style)
# 2. key=value (omegaconf style)
STEP_N=""
LOOP_N=""
TIMEOUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --step-n)
            STEP_N="$2"
            shift 2
            ;;
        --loop-n)
            LOOP_N="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        step-n=*)
            STEP_N="${1#step-n=}"
            shift
            ;;
        loop-n=*)
            LOOP_N="${1#loop-n=}"
            shift
            ;;
        timeout=*)
            TIMEOUT="${1#timeout=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Configuration:"
echo "  Competition: ${COMPETITION_ID}"
[ -n "$STEP_N" ] && echo "  Max steps: ${STEP_N}"
[ -n "$LOOP_N" ] && echo "  Max loops: ${LOOP_N}"
[ -n "$TIMEOUT" ] && echo "  Timeout: ${TIMEOUT}"
echo "  RD-Agent directory: ${RDAGENT_DIR}"
echo "  Data directory: ${DATA_DIR}"
echo ""

# ==============================================================================
# Environment Setup
# ==============================================================================
echo "Setting up environment..."

# Load environment variables from rdagent's .env file
if [ -f "${RDAGENT_ENV_FILE}" ]; then
    echo "✓ Loading environment variables from ${RDAGENT_ENV_FILE}"
    set -a  # automatically export all variables
    source "${RDAGENT_ENV_FILE}"
    set +a
else
    echo "❌ ERROR: .env not found at ${RDAGENT_ENV_FILE}"
    exit 1
fi

# Override data paths to use the workspace data directory
export DS_LOCAL_DATA_PATH="${DATA_DIR}"
export KG_LOCAL_DATA_PATH="${DATA_DIR}"
export DS_IF_USING_MLE_DATA=true
export KG_IF_USING_MLE_DATA=true

# Ensure we're using conda environment
export DS_Coder_CoSTEER_ENV_TYPE=conda
export DS_Runner_CoSTEER_ENV_TYPE=conda
export DS_Coder_CoSTEER_ENV_CONDA_ENV=medagent
export DS_Runner_CoSTEER_ENV_CONDA_ENV=medagent

# Verify data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "❌ ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

echo "✓ Data directory found: ${DATA_DIR}"

# Initialize conda
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate medagent
echo "✓ Activated conda environment: medagent"

# Verify RD-Agent is installed
if ! command -v rdagent &> /dev/null; then
    echo "❌ ERROR: rdagent command not found"
    echo "Please install RD-Agent in the medagent environment"
    exit 1
fi

echo "✓ RD-Agent installed"

# Check for API key
if [ -z "$AZURE_API_KEY" ]; then
    echo "❌ ERROR: AZURE_API_KEY not set in .env file"
    exit 1
fi

echo "✓ Azure OpenAI API key configured"

# ==============================================================================
# Patch RDAgent for Med-MLE-Bench
# ==============================================================================
echo ""
echo "Patching RDAgent for Med-MLE-Bench compatibility..."

# First, apply file-based patches to rdagent installation
echo "Applying file-based patches to RDAgent package..."
python "${AGENT_DIR}/rdagent_package_edits/apply_patches.py"
PATCH_EXIT_CODE=$?

if [ $PATCH_EXIT_CODE -ne 0 ]; then
    echo "⚠ Warning: File-based patches failed (exit code $PATCH_EXIT_CODE)"
    echo "Continuing anyway with runtime patches..."
fi

# Create a Python script to apply runtime patches before running rdagent
cat > "${AGENT_DIR}/apply_rexmle_patch.py" <<'PATCH_EOF'
#!/usr/bin/env python
"""Apply Med-MLE-Bench runtime patches to RDAgent before execution."""
import sys
import os
from pathlib import Path

# Set environment variable to indicate we're using MLE data
os.environ['DS_IF_USING_MLE_DATA'] = 'true'
os.environ['KG_IF_USING_MLE_DATA'] = 'true'

# Find med-mle-bench directory by going up from AGENT_DIR
agent_dir = Path(os.environ.get('AGENT_DIR', os.path.dirname(__file__)))
rexmle_dir = agent_dir
for _ in range(10):  # Go up max 10 levels
    rexmle_dir = rexmle_dir.parent
    if (rexmle_dir / 'rexmle').exists():
        break

os.environ['rexmle_DIR'] = str(rexmle_dir)

# Import and apply the patch
sys.path.insert(0, str(agent_dir))
from rexmle_scenario import patch_rdagent_for_rexmle
patch_rdagent_for_rexmle()

print("✓ Med-MLE-Bench runtime patches applied successfully")
PATCH_EOF

# Apply the runtime patch by running the script
python "${AGENT_DIR}/apply_rexmle_patch.py"

echo "✓ All patches applied"

# ==============================================================================
# Run RD-Agent
# ==============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                           Starting RD-Agent                                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Create sitecustomize.py to auto-apply patches when Python starts
# This ensures patches are applied before rdagent imports its modules
cat > "${AGENT_DIR}/sitecustomize.py" <<'SITECUSTOMIZE_EOF'
"""
Automatically apply Med-MLE-Bench patches when Python starts.
This file is named sitecustomize.py and will be auto-imported by Python.
"""
import os
import sys
from pathlib import Path

# Set environment variables
os.environ['DS_IF_USING_MLE_DATA'] = 'true'
os.environ['KG_IF_USING_MLE_DATA'] = 'true'

# Find med-mle-bench directory by going up from current file location
agent_dir = Path(os.path.dirname(__file__))
rexmle_dir = agent_dir
for _ in range(10):  # Go up max 10 levels
    rexmle_dir = rexmle_dir.parent
    if (rexmle_dir / 'rexmle').exists():
        break

os.environ['rexmle_DIR'] = str(rexmle_dir)

# Add current directory to path to find rexmle_scenario
sys.path.insert(0, str(agent_dir))

# Import and apply patches
try:
    from rexmle_scenario import patch_rdagent_for_rexmle
    patch_rdagent_for_rexmle()
except ImportError:
    # Silently fail if running outside of rdagent context
    pass
SITECUSTOMIZE_EOF

# Use PYTHONPATH to ensure sitecustomize.py is found
export PYTHONPATH="${AGENT_DIR}:${PYTHONPATH}"

# Build command - just use rdagent CLI directly
# sitecustomize.py will auto-apply patches when Python starts
CMD="rdagent data_science --competition ${COMPETITION_ID}"

# Add optional arguments (use hyphens, not underscores)
[ -n "$STEP_N" ] && CMD="${CMD} --step-n ${STEP_N}"
[ -n "$LOOP_N" ] && CMD="${CMD} --loop-n ${LOOP_N}"
[ -n "$TIMEOUT" ] && CMD="${CMD} --timeout ${TIMEOUT}"

echo "Command: ${CMD}"
echo ""

# Change to agent directory to run rdagent (it creates log/ directory in cwd)
cd "${AGENT_DIR}"

# Run rdagent
eval ${CMD}
EXIT_CODE=$?

# ==============================================================================
# Extract and copy submission files (similar to extract_submission_workspaces.py)
# ==============================================================================
echo ""
echo "Extracting submission files..."

# Find the latest log directory with sota_exp_to_submit by iterating loops (newest first)
# Pattern: log/<TIMESTAMP>/Loop_<N>/record/sota_exp_to_submit/<PID>/<TIMESTAMP>.pkl
WORKSPACE_HASH=""
LATEST_LOG=""
HASH_FOUND=false

# Find all log session directories (sorted newest first)
for LOG_SESSION_DIR in $(find log -maxdepth 1 -type d -name "20*" 2>/dev/null | sort -r); do
    echo "Debug: Checking session directory: $LOG_SESSION_DIR"

    # Find all loop directories in this session, sorted by loop number (newest first)
    if [ -d "$LOG_SESSION_DIR" ]; then
        for LOOP_DIR in $(ls -d "${LOG_SESSION_DIR}/Loop_"* 2>/dev/null | sort -V -r); do
            SOTA_DIR="${LOOP_DIR}/record/sota_exp_to_submit"

            if [ -d "$SOTA_DIR" ]; then
                echo "Debug: Found sota_exp_to_submit in: $SOTA_DIR"

                # Find the latest pickle file in this loop's sota directory
                LATEST_PKL=$(find "$SOTA_DIR" -type f -name "*.pkl" 2>/dev/null | xargs -r ls -t | head -1)

                if [ -n "$LATEST_PKL" ]; then
                    LATEST_LOG=$(dirname "$LATEST_PKL")
                    echo "Debug: Found latest pkl in loop: $LATEST_PKL"
                    echo "Debug: LATEST_LOG = $LATEST_LOG"

                    # List what's in the directory for debugging
                    echo "Contents of ${LATEST_LOG}:"
                    ls -la "${LATEST_LOG}" || true

                    # Extract workspace hash from the pickle file
                    for pkl_file in "${LATEST_LOG}"/*.pkl; do
                        if [ -f "$pkl_file" ]; then
                            echo "Debug: Checking pickle file: $pkl_file"
                            # Extract hash from pickle (32-char hex string after RD-Agent_workspace)
                            HASH=$(strings "$pkl_file" 2>/dev/null | grep -oP 'RD-Agent_workspace/\K[a-f0-9]{32}' | head -1)
                            if [ -n "$HASH" ]; then
                                echo "Debug: Found workspace hash: $HASH"
                                WORKSPACE_HASH="$HASH"
                                # Break out of all loops using goto-like behavior (using label)
                                break
                            fi
                        fi
                    done

                    # If we found a hash, check if submission.csv exists
                    if [ -n "$WORKSPACE_HASH" ]; then
                        # Construct workspace path
                        WORKSPACE_PATH="${AGENT_DIR}/git_ignore_folder/RD-Agent_workspace/${WORKSPACE_HASH}"
                        echo "Found workspace: ${WORKSPACE_PATH}"

                        # Check if submission.csv exists in this workspace
                        if [ -f "${WORKSPACE_PATH}/submission.csv" ]; then
                            echo "✓ submission.csv found in workspace, copying entire hash folder contents"
                            HASH_FOUND=true
                            break 2  # Break out of both loop iterations
                        else
                            echo "⚠ submission.csv not found in ${WORKSPACE_PATH}, continuing to previous loops..."
                            WORKSPACE_HASH=""  # Reset to try next loop
                        fi
                    fi
                fi
            fi
        done
    fi
done

if [ "$HASH_FOUND" = true ] && [ -n "$WORKSPACE_HASH" ]; then
    # Construct workspace path
    WORKSPACE_PATH="${AGENT_DIR}/git_ignore_folder/RD-Agent_workspace/${WORKSPACE_HASH}"

    if [ -d "$WORKSPACE_PATH" ]; then
        # Copy entire hash folder contents to submission directory (excluding symlinks to preserve structure)
        echo "Copying entire workspace contents from ${WORKSPACE_PATH} to ${SUBMISSION_DIR}/"
        cp -r "${WORKSPACE_PATH}"/* "${SUBMISSION_DIR}/" 2>/dev/null || true
        echo "✓ Copied all workspace contents to submission directory"

        # Copy the entire workspace code for reference
        mkdir -p "${CODE_DIR}/workspace"
        cp -r "${WORKSPACE_PATH}"/*.py "${CODE_DIR}/workspace/" 2>/dev/null || true
        echo "✓ Copied workspace code to ${CODE_DIR}/workspace/"
    else
        echo "⚠ Warning: Workspace directory not found at ${WORKSPACE_PATH}"
    fi

    # Copy all logs
    SESSION_LOG=$(ls -td log/*/__session__/* 2>/dev/null | head -1)
    if [ -n "$SESSION_LOG" ]; then
        cp -r "${SESSION_LOG}" "${LOGS_DIR}/rdagent_session" 2>/dev/null || true
        echo "✓ Copied session logs to ${LOGS_DIR}/"
    fi

    echo ""
    echo "Submission folder contents:"
    ls -laR "${SUBMISSION_DIR}/" || true
else
    echo "⚠ Warning: Could not find workspace with submission.csv"
    echo "Checking for session logs as fallback..."

    # Fallback to finding submission in session directory
    SESSION_LOG=$(ls -td log/*/__session__/* 2>/dev/null | head -1)
    if [ -n "$SESSION_LOG" ] && [ -d "${SESSION_LOG}/ckpt" ]; then
        find "${SESSION_LOG}/ckpt" -name "submission.csv" -exec cp {} "${SUBMISSION_DIR}/" \; 2>/dev/null || true
        echo "Copied submission.csv from session checkpoint"
    fi
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                           Completed                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ RD-Agent completed successfully"
else
    echo "✗ RD-Agent failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
