"""
ReX-MLE specific scenario for RDAgent.
This module patches RDAgent's Kaggle API calls to use local ReX-MLE data instead.
"""

import json
import os
from pathlib import Path
from typing import List

import pandas as pd


def get_rexmle_challenge_dir(competition: str) -> Path:
    """
    Get the ReX-MLE challenge directory for a competition.

    Args:
        competition: Competition ID (e.g., "dentex")

    Returns:
        Path to the challenge directory
    """
    # Try to find the rexmle challenges directory
    # Look relative to this file's location
    current_file = Path(__file__).resolve()

    # Try going up to find rex-mle directory
    search_dir = current_file
    for _ in range(10):  # Max 10 levels up
        search_dir = search_dir.parent
        challenge_dir = search_dir / "rexmle" / "challenges" / competition
        if challenge_dir.exists():
            return challenge_dir

    # Fallback: use environment variable or absolute path
    base_dir = os.getenv("rexmle_DIR", "SET_PATH")
    challenge_dir = Path(base_dir) / "rexmle" / "challenges" / competition

    if not challenge_dir.exists():
        raise FileNotFoundError(f"Cannot find ReX-MLE challenge directory for {competition}")

    return challenge_dir


def leaderboard_scores(competition: str) -> List[float]:
    """
    Get leaderboard scores from ReX-MLE local data instead of Kaggle API.

    This function replaces rdagent's kaggle_crawler.leaderboard_scores() to work
    with ReX-MLE challenges.

    Args:
        competition: Competition ID

    Returns:
        List of scores from the leaderboard (sorted by rank)
    """
    challenge_dir = get_rexmle_challenge_dir(competition)
    leaderboard_file = challenge_dir / "leaderboard.csv"

    if not leaderboard_file.exists():
        # If no leaderboard file, return default scores
        # Assume higher is better with some sample scores
        return [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # Read leaderboard CSV
    df = pd.read_csv(leaderboard_file)

    # Find the primary metric (usually the first metric column after team/rank)
    # Look for metric_config.json to determine primary metric
    metric_config_file = challenge_dir / "metric_config.json"
    if metric_config_file.exists():
        with open(metric_config_file) as f:
            metric_config = json.load(f)

        # Get the first metric as primary metric
        metric_directions = metric_config.get("metric_directions", {})
        if metric_directions:
            primary_metric = list(metric_directions.keys())[0]
        else:
            # Fallback: find first numeric column that's not rank
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            primary_metric = [col for col in numeric_cols if col != 'rank'][0]
    else:
        # Fallback: find mean_position or first numeric column
        if 'mean_position' in df.columns:
            # mean_position is the average rank, lower is better
            # Return sorted scores (worst to best for mean_position)
            return sorted(df['mean_position'].tolist(), reverse=True)

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        primary_metric = [col for col in numeric_cols if col != 'rank'][0]

    # Extract scores for the primary metric
    if primary_metric in df.columns:
        scores = df[primary_metric].tolist()
        return scores

    # Fallback if no suitable metric found
    return [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def get_evaluation_metric_direction(competition: str) -> bool:
    """
    Determine if higher scores are better for this competition.

    Args:
        competition: Competition ID

    Returns:
        True if higher is better, False if lower is better
    """
    challenge_dir = get_rexmle_challenge_dir(competition)
    metric_config_file = challenge_dir / "metric_config.json"

    if metric_config_file.exists():
        with open(metric_config_file) as f:
            metric_config = json.load(f)

        metric_directions = metric_config.get("metric_directions", {})
        if metric_directions:
            # Get the first metric direction
            first_direction = list(metric_directions.values())[0]
            return first_direction == "higher_better"

    # Default: assume higher is better
    return True


def patch_rdagent_for_rexmle():
    """
    Monkey-patch RDAgent's Kaggle API functions to use ReX-MLE data.

    This should be called at the start of rdagent execution when running
    with ReX-MLE competitions.
    """
    try:
        # Patch the kaggle_crawler module
        from rdagent.scenarios.kaggle import kaggle_crawler

        # Replace the leaderboard_scores function
        kaggle_crawler.leaderboard_scores = leaderboard_scores

        print("✓ Successfully patched rdagent.scenarios.kaggle.kaggle_crawler.leaderboard_scores")

    except ImportError as e:
        print(f"⚠ Warning: Could not patch rdagent kaggle_crawler: {e}")
        print("  RDAgent may not work correctly with ReX-MLE competitions")

    # Apply embedding truncation fix to prevent crashes on long code files
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__) + "/rdagent_package_edits")
        from embedding_truncation_fix import patch_embedding_truncation
        patch_embedding_truncation()
    except Exception as e:
        print(f"⚠ Warning: Could not apply embedding truncation fix: {e}")
        print("  RDAgent may crash when code files exceed embedding token limits")

    # Patch get_ds_env to handle conda/LocalEnv mode
    # This prevents permission errors when trying to create /mle/data symlinks
    try:
        from rdagent.components.coder.data_science import conf as ds_conf
        import inspect

        _original_get_ds_env = ds_conf.get_ds_env

        def patched_get_ds_env(**kwargs):
            # Check if we're using conda/LocalEnv mode
            try:
                from rdagent.components.coder.data_science.conf import DSCoderCoSTEERSettings
                conf = DSCoderCoSTEERSettings()

                if conf.env_type == "conda":
                    # For LocalEnv (conda mode), we need to manually create workspace_input symlinks
                    # because LocalEnv doesn't support Docker-style volume mapping
                    extra_volumes = kwargs.get('extra_volumes', {})

                    # Filter out extra_volumes from kwargs (LocalEnv doesn't accept it)
                    kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'extra_volumes'}
                    env = _original_get_ds_env(**kwargs_filtered)

                    # After creating the env, we'll create workspace_input symlinks in a hook
                    if extra_volumes:
                        # Store the volume mappings for later use
                        if not hasattr(env, '_rexmle_extra_volumes'):
                            env._rexmle_extra_volumes = {}
                        env._rexmle_extra_volumes.update(extra_volumes)

                        # Add a hook to create symlinks before running
                        original_run = env.run
                        def run_with_symlinks(*args, **kwargs_run):
                            # Get the working directory from the local_path parameter passed to run()
                            # local_path is the second positional arg or can be passed as keyword
                            # It defaults to "." which is the workspace directory where code executes
                            from pathlib import Path
                            
                            # Extract local_path from args or kwargs
                            # Signature: run(entry=None, local_path=".", env=None, **kwargs)
                            if len(args) >= 2:
                                local_path = args[1]
                            elif 'local_path' in kwargs_run:
                                local_path = kwargs_run['local_path']
                            else:
                                local_path = "."  # Default from function signature
                            
                            # Convert to absolute path - this is the workspace directory where code executes
                            exec_dir = Path(local_path).resolve()
                            
                            # Create workspace_input symlinks for each volume mapping
                            for source_path, target_path in env._rexmle_extra_volumes.items():
                                # target_path is like "./workspace_input/" or "workspace_input"
                                # Remove leading "./" and trailing "/" if present
                                target_clean = target_path.lstrip('./').rstrip('/')
                                            
                                # Create symlink in the execution directory (workspace)
                                workspace_input = exec_dir / target_clean
                                
                                # Resolve source path to absolute
                                source_abs = Path(source_path).resolve()
                                
                                # Check if source exists
                                if not source_abs.exists():
                                    print(f"⚠ Warning: Source path does not exist: {source_abs}")
                                    continue
                                
                                # Check if symlink already exists and is valid
                                if workspace_input.exists():
                                    if workspace_input.is_symlink():
                                        # Check if existing symlink points to correct location
                                        existing_target = workspace_input.resolve()
                                        if existing_target == source_abs:
                                            # Symlink already exists and points to correct location
                                            continue
                                        else:
                                            # Remove incorrect symlink
                                            workspace_input.unlink()
                                    else:
                                        # Path exists but is not a symlink - skip to avoid overwriting
                                        print(f"⚠ Warning: Path exists but is not a symlink: {workspace_input}")
                                        continue
                                
                                try:
                                    # Create parent directory if needed
                                    workspace_input.parent.mkdir(parents=True, exist_ok=True)
                                    # Create symlink from workspace_input to source_path (absolute)
                                    workspace_input.symlink_to(source_abs)
                                    print(f"✓ Created workspace_input symlink: {workspace_input} -> {source_abs}")
                                except Exception as e:
                                    print(f"⚠ Warning: Could not create workspace_input symlink: {e}")
                                    import traceback
                                    traceback.print_exc()

                            # Now call the original run() with the same arguments
                            return original_run(*args, **kwargs_run)

                        env.run = run_with_symlinks

                    return env
            except Exception:
                pass  # If we can't check, just call original with all params

            # Get the signature of the original function to filter kwargs
            sig = inspect.signature(_original_get_ds_env)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return _original_get_ds_env(**valid_kwargs)

        ds_conf.get_ds_env = patched_get_ds_env
        print("✓ Successfully patched rdagent.components.coder.data_science.conf.get_ds_env")

    except Exception as e:
        print(f"⚠ Warning: Could not patch get_ds_env: {e}")
        print("  May encounter permission errors with /mle/data symlinks")


# Auto-patch if DS_IF_USING_MLE_DATA or KG_IF_USING_MLE_DATA is set
if __name__ != "__main__":
    if os.getenv("DS_IF_USING_MLE_DATA", "").lower() in ["true", "1", "yes"] or \
       os.getenv("KG_IF_USING_MLE_DATA", "").lower() in ["true", "1", "yes"]:
        patch_rdagent_for_rexmle()
