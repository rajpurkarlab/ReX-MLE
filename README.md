# ReX-MLE

A medical machine learning benchmark platform for evaluating automated machine learning agents on realistic healthcare tasks.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
  - [Challenge Preparation](#challenge-preparation)
  - [Running Agents](#running-agents)
  - [Grading Submissions](#grading-submissions)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

## Overview

TrialReX-MLE provides a framework for:
- Running multiple ML agents (RD-Agent, ML-Master, etc.) on standardized medical ML challenges
- Preparing and managing challenge datasets
- Evaluating agent submissions against benchmark metrics
- Analyzing agent strategies and performance

## Installation

### Prerequisites

- Miniconda or Anaconda installed
- Bash shell
- Python 3.11+

### Initial Setup

Run the setup script to create the required conda environments:

```bash
./setup.sh
```

This installs:
- `rexmle`: The evaluator environment for challenge management and grading
- `rexagent`: The base agent environment for running agents


## Setup

After running `./setup.sh`, the conda environments are ready to use. Make sure both `rexmle` and `rexagent` environments are properly installed before proceeding.

## Usage

### Challenge Preparation

Before running agents on a challenge, you need to prepare the challenge data.

1. Activate the rexmle environment:
```bash
conda activate rexmle
```

2. Change to the rex-mle directory:
```bash
cd ./rex-mle
```

3. List available challenges:
```bash
python -m rexmle.cli list
```

4. View challenge information:
```bash
python -m rexmle.cli info CHALLENGE_NAME
```

5. Prepare the challenge:
```bash
python -m rexmle.cli prepare CHALLENGE_NAME
```

### Agent-Specific Setup


#### ML-Master Setup

For ML-Master agent, install additional dependencies:

```bash
bash setup/setup_mlmaster.sh
```

#### RD-Agent Data Setup

After preparing challenges, setup RD-Agent data directory with symlinks to challenge data:

```bash
cd rex-mle
python setup_rdagent_data.py
```

### Running Agents

To run agents, use the scripts in `rex-mle/` (e.g., `run_aide.sh`, `run_mlmaster.sh`, `run_rdagent.sh`). Each script supports configurable model variants and time limits. All scripts assume you are already in a GPU-enabled compute environment.

Environment variables (including API credentials) should be set in a `.env` file in the project root before running agents.

#### Creating Custom Agents

To run your own agent, create a similar folder in `rex-mle/agents/` and implement a startup script (e.g., `run_agent_*.py`). Follow the pattern of existing agents (AIDE, ML-Master, RD-Agent) for consistency with the evaluation framework.

### Grading Submissions

Once an agent completes and generates a submission, you can grade the results.

1. Create a JSONL file listing submission paths (see `example_submission.jsonl` for format):
```json
{"submission_dir": "/path/to/submission/directory"}
```

2. Grade the submissions:
```bash
cd rex-mle
python -m rexmle.cli grade-batch --submission ./your_submission.jsonl --output-dir ./metrics --suffix your_suffix
```

The grading output will be saved to the specified output directory with evaluation metrics.

## Project Structure

```
TrialReX-MLE/
├── setup.sh                    # Main setup script
├── setup/                      # Setup scripts for specific components
│   └── setup_mlmaster.sh      # ML-Master specific setup
├── rex-mle/                   # Core evaluation and challenge management
│   ├── rexmle/               # ReX-MLE package
│   ├── agents/               # Agent implementations
│   │   ├── rdagent/         # RD-Agent configuration
│   │   ├── ml-master/       # ML-Master configuration
│   │   └── ...              # Other agents
│   ├── challenges/           # Challenge definitions and data
│   └── example_submission.jsonl
├── strategies/               # Strategy analysis and documentation
└── README.md

```

## Documentation

### Strategy Scoring

After grading submissions, you can score agent logs for the 13 challenge strategies using the `/strategies` folder. Each agent (AIDE, MLMaster, RDAgent) has its own preprocessing pipeline:

```bash
cd strategies/
python analyze_strategies.py --batch-dir <preprocessed-logs>
python aggregate_strategy_scores.py --scores-dir <scores-dir> --output <output>.json
```

See [strategies/README.md](strategies/README.md) for detailed instructions for each agent type.

### Agent Configuration

Each agent directory contains its own documentation for specific configuration and usage.

## Notes

- Ensure you have sufficient disk space for challenge data and agent outputs
- Some challenges may require significant computational resources
- Check individual agent directories for specific requirements and troubleshooting
