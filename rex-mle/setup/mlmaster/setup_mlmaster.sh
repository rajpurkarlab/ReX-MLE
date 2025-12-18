#!/bin/bash
# Setup script for ML-Master agent dependencies
# Installs all required packages for ML-Master from requirements_mlmaster.txt

set -e

echo "========================================"
echo "Setting up ML-Master environment"
echo "========================================"
echo ""

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the environment
echo "Activating rexagent environment..."
conda activate rexagent

# Install ML-Master dependencies (without version pinning, only if not already installed)
echo ""
echo "Installing ML-Master dependencies (flexible versions, skipping already installed)..."

# Extract package names from requirements file (remove version specifiers)
# This allows pip to use any compatible version and skip already installed packages
grep -v '^#' requirements_mlmaster.txt | grep -v '^$' | cut -d'=' -f1 | xargs pip install --upgrade-strategy only-if-needed 2>&1 | grep -E "(Successfully|already satisfied|Collecting)" | tail -20

echo ""
echo "✓ Dependency installation complete (only new packages were installed)"

echo ""
echo "========================================"
echo "✓ ML-Master setup complete!"
echo "========================================"
echo ""
echo "ML-Master dependencies have been installed."
echo "To run ML-Master agent:"
echo "  python run_agent_no_docker.py --agent-id mlmaster --competition-set experiments/splits/ldct-iqa.txt"
echo ""
