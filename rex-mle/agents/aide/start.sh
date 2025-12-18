#!/bin/bash
set -x # Print commands and their arguments as they are executed

cd ${AGENT_DIR}

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate rexagent

# determine hardware available
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE
# check that we can use the GPU in PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"
# check that we can use the GPU in TensorFlow
python -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))"

# convert $TIME_LIMIT_SECS to more readable format for prompt
# Only set default if not already set by environment (e.g., from config file)
: ${TIME_LIMIT_SECS:=43200}
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# overwrite instructions.txt with instructions_obfuscated.txt if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  INSTRUCTIONS_DIR=$(dirname ${INSTRUCTIONS_FILE})
  if [ ! -w ${INSTRUCTIONS_DIR} ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv ${INSTRUCTIONS_DIR}/instructions_obfuscated.txt ${INSTRUCTIONS_FILE}
fi

# start a new file to store the full instructions, starting with general instructions
cp ${INSTRUCTIONS_FILE} ${AGENT_DIR}/full_instructions.txt

# Update instructions for agent-specific details: replace `/home/` paths to make paths relative
# (since the agent will have its own copies of these files in its workspace).
# e.g. /home/submission/submission.csv -> submission/submission.csv
sed -i 's|/home/||g' ${AGENT_DIR}/full_instructions.txt
# we'll take care of moving things to home/submission/ ourselves

# move on to agent-specific instructions, with a linebreak in between
# substitute env variables into additional_notes.txt and append result to full_instructions.txt
echo "" >> ${AGENT_DIR}/full_instructions.txt
envsubst < ${AGENT_DIR}/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt
# finally, append the comp instructions, with a linebreak in between
printf "\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt

# overwrite description.md with description_obfuscated.md if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w ${DATA_DIR} ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv ${DATA_DIR}/description_obfuscated.md ${DATA_DIR}/description.md
fi
cat ${DATA_DIR}/description.md >> ${AGENT_DIR}/full_instructions.txt

# append grading script if it exists
if [ -f "${DATA_DIR}/grade.py" ]; then
  printf "\n\nGRADING SCRIPT\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt
  cat ${DATA_DIR}/grade.py >> ${AGENT_DIR}/full_instructions.txt
  echo "Appended grade.py to instructions"
fi

# Don't create directories in advance - let AIDE create them
# Otherwise AIDE will see existing dirs and increment the index (0-exp -> 1-exp -> 2-exp)

# run with timeout, and print if timeout occurs
timeout $TIME_LIMIT_SECS aide data_dir="${DATA_DIR}" desc_file="${AGENT_DIR}/full_instructions.txt" \
  exp_name="exp" \
  $@ # forward the bash arguments to aide
AIDE_EXIT=$?

if [ $AIDE_EXIT -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi

# Find which workspace AIDE actually created (should be 0-exp)
ACTUAL_WORKSPACE=$(find ${AGENT_DIR}/workspaces -maxdepth 1 -type d -name "*-exp" | head -1)
ACTUAL_LOGS=$(find ${AGENT_DIR}/logs -maxdepth 1 -type d -name "*-exp" | head -1)

if [ -n "$ACTUAL_WORKSPACE" ]; then
  WORKSPACE_NAME=$(basename "$ACTUAL_WORKSPACE")
  echo "AIDE created workspace: ${WORKSPACE_NAME}"

  # Copy entire submission folder (includes CSV and any prediction files)
  if [ -d "${ACTUAL_WORKSPACE}/submission" ]; then
    # Copy all contents from AIDE's submission folder to the target submission folder
    cp -r "${ACTUAL_WORKSPACE}/submission/"* "${SUBMISSION_DIR}/" 2>/dev/null || true
    echo "Copied submission folder contents"

    # List what was copied
    echo "Submission folder contents:"
    ls -la "${SUBMISSION_DIR}/"
  elif [ -f "${ACTUAL_WORKSPACE}/working/submission.csv" ]; then
    # Fallback: copy from working directory if no submission folder
    cp "${ACTUAL_WORKSPACE}/working/submission.csv" "${SUBMISSION_DIR}/"
    echo "Copied submission.csv from working/"
  fi
fi

if [ -n "$ACTUAL_LOGS" ]; then
  LOGS_NAME=$(basename "$ACTUAL_LOGS")
  echo "AIDE created logs: ${LOGS_NAME}"

  # Copy best solution
  if [ -f "${ACTUAL_LOGS}/best_solution.py" ]; then
    cp "${ACTUAL_LOGS}/best_solution.py" "${CODE_DIR}/"
    echo "Copied best_solution.py"
  fi
fi

exit $AIDE_EXIT
