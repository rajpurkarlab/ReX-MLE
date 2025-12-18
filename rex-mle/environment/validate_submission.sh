#!/bin/bash

# This script is a tool for agents to submit files to the grading server for validation.
# It packages the submission.csv and any associated files (e.g., predictions folder)
# into a temporary directory structure that mimics the expected submission format.

# validate_submission.sh
# Usage: ./validate_submission.sh [submission_file]
# If no submission_file is provided, defaults to submission.csv in current directory

SUBMISSION_FILE="${1:-submission.csv}"

# Read the dynamic port number from the .grading_port file
GRADING_PORT_FILE="${WORKSPACE_DIR:-$(pwd)}/.grading_port"
if [ -f "$GRADING_PORT_FILE" ]; then
    GRADING_PORT=$(cat "$GRADING_PORT_FILE")
else
    # Fallback to default port if file doesn't exist
    GRADING_PORT=5000
fi

SERVER_URL="http://localhost:${GRADING_PORT}/validate"

# Check if submission file exists
if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "Error: Submission file $SUBMISSION_FILE does not exist."
    exit 1
fi

# Get the directory containing the submission file
SUBMISSION_DIR=$(dirname "$SUBMISSION_FILE")
SUBMISSION_BASENAME=$(basename "$SUBMISSION_FILE")

# Create a temporary directory for packaging the submission
TEMP_DIR=$(mktemp -d)
TEMP_SUBMISSION_DIR="$TEMP_DIR/submission"
mkdir -p "$TEMP_SUBMISSION_DIR"

# Copy submission.csv to temp directory
cp "$SUBMISSION_FILE" "$TEMP_SUBMISSION_DIR/$SUBMISSION_BASENAME"

# Check for and copy associated files/directories
# Common patterns: predictions/, outputs/, images/, etc.
if [ -d "$SUBMISSION_DIR/predictions" ]; then
    echo "Found predictions directory, copying..."
    cp -r "$SUBMISSION_DIR/predictions" "$TEMP_SUBMISSION_DIR/"
fi

if [ -d "$SUBMISSION_DIR/outputs" ]; then
    echo "Found outputs directory, copying..."
    cp -r "$SUBMISSION_DIR/outputs" "$TEMP_SUBMISSION_DIR/"
fi

if [ -d "$SUBMISSION_DIR/images" ]; then
    echo "Found images directory, copying..."
    cp -r "$SUBMISSION_DIR/images" "$TEMP_SUBMISSION_DIR/"
fi

echo "Submitting to grading server at $SERVER_URL..."
echo "Submission contents:"
ls -lah "$SUBMISSION_DIR"

# Build curl command with submission file
CURL_CMD="curl -s -X POST -F \"file=@${SUBMISSION_FILE}\""

# If predictions folder exists, create a zip and send it too
if [ -d "$SUBMISSION_DIR/predictions" ]; then
    PREDICTIONS_ZIP="$TEMP_DIR/predictions.zip"
    echo "Creating predictions.zip..."
    (cd "$SUBMISSION_DIR" && zip -r "$PREDICTIONS_ZIP" predictions/)
    CURL_CMD="$CURL_CMD -F \"predictions_zip=@${PREDICTIONS_ZIP}\""
fi

# Execute curl command
CURL_CMD="$CURL_CMD ${SERVER_URL}"
RESPONSE=$(eval $CURL_CMD)

# Clean up temporary directory
rm -rf "$TEMP_DIR"

# Display response
echo "$RESPONSE"

# Check if validation was successful
# Handle both "is_valid": true and "is_valid":true (with/without space)
if echo "$RESPONSE" | grep -qE '"is_valid":\s*true'; then
    echo ""
    echo "✓ Submission is valid!"
    exit 0
elif echo "$RESPONSE" | grep -qE '"is_valid":\s*false'; then
    echo ""
    echo "✗ Submission validation failed."
    exit 1
else
    echo ""
    echo "⚠ Unable to determine validation status from server response."
    exit 2
fi