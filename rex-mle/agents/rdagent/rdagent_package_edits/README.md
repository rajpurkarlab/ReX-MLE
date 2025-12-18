# RD-Agent Package Edits for No-Docker Mode

This directory contains the patches that need to be applied to the installed rdagent package to enable no-docker mode in Med-MLE-Bench.

## Overview

RD-Agent by default requires Docker to run. These patches modify the installed rdagent package to support running in a conda environment without Docker when the appropriate environment variables are set.

## Environment Variables

The patches check for these environment variables (set by `start.sh`):
- `DS_Coder_CoSTEER_ENV_TYPE=conda` (or `local`)
- `DS_Runner_CoSTEER_ENV_TYPE=conda` (or `local`)

When these are set, RD-Agent will:
1. Skip Docker client initialization
2. Skip Docker container operations
3. Expect data to be pre-prepared in the local filesystem

## Files to Patch

### 1. `rdagent/utils/env.py`
**Location:** `<conda_env>/lib/python3.11/site-packages/rdagent/utils/env.py`

**Changes:**
- `DockerEnv.prepare()`: Add check to skip Docker operations in conda/local mode
- `DockerEnv._run()`: Add check to prevent Docker container runs in conda/local mode

**Details:** See `env.txt`

### 2. `rdagent/scenarios/kaggle/kaggle_crawler.py`
**Location:** `<conda_env>/lib/python3.11/site-packages/rdagent/scenarios/kaggle/kaggle_crawler.py`

**Changes:**
- `download_data()`: Add no-docker mode detection and skip Docker-based data downloads
- Modified `competition_local_path` to point directly to `local_path` in no-docker mode (since data is symlinked directly, not in a subdirectory)

**Details:** See `kaggle_crawler.txt`

### 3. `rdagent/components/coder/data_science/conf.py`
**Location:** `<conda_env>/lib/python3.11/site-packages/rdagent/components/coder/data_science/conf.py`

**Changes:**
- Add `env_conda_env` field to `DSCoderCoSTEERSettings` (defaults to "rexagent")
- Update `get_ds_env()` to use `conf.env_conda_env` instead of `conf_type` for conda environment name

**Details:** See `coder_conf.txt`

### 4. `rdagent/scenarios/data_science/dev/runner/__init__.py`
**Location:** `<conda_env>/lib/python3.11/site-packages/rdagent/scenarios/data_science/dev/runner/__init__.py`

**Changes:**
- Add `env_conda_env` field to `DSRunnerCoSTEERSettings` (defaults to "rexagent")

**Details:** See `runner_init.txt`

## How to Apply Patches

### Automatic Application
You can apply these patches by directly editing the files in the conda environment:

```bash
conda activate rexagent

# Find the rdagent installation directory
RDAGENT_DIR=$(python -c "import rdagent; import os; print(os.path.dirname(rdagent.__file__))")

# Apply patches manually using the instructions in env.txt and kaggle_crawler.txt
```

### Manual Application
1. Open each file listed in the patch files
2. Search for the "SEARCH FOR" text
3. Replace with the "REPLACE WITH" text
4. Save the file

## Verification

After applying patches, verify they work by:

```bash
# Set environment variables
export DS_Coder_CoSTEER_ENV_TYPE=conda
export DS_Runner_CoSTEER_ENV_TYPE=conda

# Test that rdagent can import without errors
python -c "from rdagent.utils.env import MLEBDockerEnv; env = MLEBDockerEnv(); env.prepare()"
# Should print: "Docker prepare() called but conda/local mode is enabled - skipping"
```

## Data Preparation

In no-docker mode, competition data must be pre-prepared using `rexmlebench prepare`:

```bash
rexmlebench prepare -c <competition_id>
```

This ensures the data is available at the expected location before running RD-Agent.

## Troubleshooting

### "Docker connection error"
- Ensure environment variables are set before running rdagent
- Verify patches were applied correctly to the installed package (not just the local copy)

### "Competition data not found"
- Run `rexmlebench prepare -c <competition_id>` before running the agent
- Verify `DS_LOCAL_DATA_PATH` points to the correct location

### "ImportError" or syntax errors
- Check that all patches were applied completely
- Ensure indentation is correct (Python is whitespace-sensitive)

## Notes

- These patches modify the **installed** rdagent package in the conda environment
- The patches are also applied to the local copy in `med-mle-bench/agents/rdagent/` for consistency
- If you reinstall rdagent (e.g., via `pip install --upgrade rdagent`), you'll need to reapply these patches
