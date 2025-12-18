#!/bin/bash
# Setup script for running Med-MLE-bench agents without Docker
# This script creates the necessary conda environments for running agents locally

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "ReX-MLE Setup"
echo "=================================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

echo "Step 1: Creating 'rexagent' conda environment from rexagent.yml..."
if conda env list | grep -q "^rexagent "; then
    echo "  Environment 'rexagent' already exists. Skipping..."
else
    conda env create -f rexagent.yml
fi

echo "  Installing rdagent==0.8.0..."
conda run -n rexagent pip install rdagent==0.8.0 --upgrade
echo "  Installing aideml==0.2.2..."
conda run -n rexagent pip install aideml==0.2.2 --upgrade

echo ""
echo "Step 2: Creating 'rexmle' conda environment from rexmle.yml..."
if conda env list | grep -q "^rexmle "; then
    echo "  Environment 'rexmle' already exists. Skipping..."
else
    conda env create -f rexmle.yml
fi

echo "  Installing rdagent==0.8.0..."
conda run -n rexmle pip install rdagent==0.8.0 --upgrade
echo "  Installing aideml==0.2.2..."
conda run -n rexmle pip install aideml==0.2.2 --upgrade

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Conda environments created:"
echo "  - 'rexagent': For running ML agents with all dependencies"
echo "  - 'rexmle': For grading server and rexmle tools"
echo ""