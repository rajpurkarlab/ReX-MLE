"""
Utility functions for rex-mle.
"""

import logging
import sys
import yaml
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Dict
import importlib
import os


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def get_module_dir() -> Path:
    """
    Get the directory containing the rexmle package.

    Returns:
        Path to rexmle directory
    """
    return Path(__file__).parent


def get_repo_dir() -> Path:
    """
    Get the repository root directory.

    Returns:
        Path to repository root
    """
    return Path(__file__).parent.parent


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """
    Load a YAML file.

    Args:
        filepath: Path to YAML file

    Returns:
        Dictionary of YAML contents

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not filepath.exists():
        raise FileNotFoundError(f"YAML file not found: {filepath}")

    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save a dictionary to a YAML file.

    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def read_csv(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments to pass to pd.read_csv()

    Returns:
        DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    return pd.read_csv(filepath, **kwargs)


def import_fn(module_path: str) -> Callable:
    """
    Import a function from a module path string.

    Args:
        module_path: String in format "module.path:function_name"
                     e.g., "rexmle.challenges.challenge1.grade:grade"

    Returns:
        Imported function

    Raises:
        ValueError: If module_path format is invalid
        ImportError: If module or function not found
    """
    if ':' not in module_path:
        raise ValueError(
            f"Invalid module path format: {module_path}. "
            "Expected format: 'module.path:function_name'"
        )

    module_name, function_name = module_path.rsplit(':', 1)

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")

    if not hasattr(module, function_name):
        raise ImportError(
            f"Module '{module_name}' has no function '{function_name}'"
        )

    return getattr(module, function_name)


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Get a timestamp string for naming runs/directories.

    Returns:
        Timestamp in YYYYMMDD_HHMMSS format
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_runs_dir() -> Path:
    """
    Get the runs directory where agent runs are stored.

    Returns:
        Path to runs directory
    """
    env_override = os.getenv("RUNS_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return get_repo_dir() / "runs"


def create_run_dir(challenge_id: str, agent_id: str, run_group: str = None) -> Path:
    """
    Create a new run directory for storing agent outputs.

    Args:
        challenge_id: Challenge identifier
        agent_id: Agent identifier
        run_group: Optional run group name

    Returns:
        Path to created run directory
    """
    runs_dir = get_runs_dir()
    runs_dir.mkdir(parents=True, exist_ok=True)

    if run_group:
        run_dir = runs_dir / run_group / f"{get_timestamp()}_{challenge_id}_{agent_id}"
    else:
        run_dir = runs_dir / f"{get_timestamp()}_{challenge_id}_{agent_id}"

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def is_dataset_prepared(challenge) -> bool:
    """
    Check if a challenge's dataset has been prepared.

    Args:
        challenge: Challenge instance

    Returns:
        True if dataset is prepared (public dir exists and has files)
    """
    if not challenge.public_dir.exists():
        return False

    # Check if public dir has any files
    files = list(challenge.public_dir.rglob("*"))
    return len(files) > 0


def purple(text: str) -> str:
    """
    Return text colored purple for terminal output.

    Args:
        text: Text to color

    Returns:
        Text wrapped in ANSI purple color codes
    """
    return f"\033[95m{text}\033[0m"
