#!/bin/bash
set -x # Print commands and their arguments as they are executed

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate rexagent

python ${AGENT_DIR}/main.py

# Check that the agent doesn't have permissions to read private dir
ls /private || echo "Cannot access /private (expected)"

# Check that the agent does have permissions to read/write everything in workspace
ls ${DATA_DIR}
touch $CODE_DIR/code.py
touch $LOGS_DIR/run.log
touch $AGENT_DIR/agent_was_here.txt
cat ${INSTRUCTIONS_FILE}

# Use the environment-provided grading server to validate our submission
bash ${VALIDATION_SCRIPT} ${SUBMISSION_DIR}/submission.csv
