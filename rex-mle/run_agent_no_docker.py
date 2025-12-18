#!/usr/bin/env python
"""
Agent runner that works without Docker.
This script runs agents locally using conda environments instead of containers.
"""
import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.registry import Agent
from agents.registry import registry as agent_registry
from grading_server_standalone import GradingServerManager
from rexmle.registry import Challenge as Competition, registry
from rexmle.utils import create_run_dir, get_logger, get_runs_dir, get_timestamp, is_dataset_prepared

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = get_logger(__name__)


@dataclass(frozen=True)
class Task:
    run_id: str
    seed: int
    path_to_run_group: Path
    path_to_run: Path
    agent: Agent
    competition: Competition


def setup_agent_workspace(
    competition: Competition,
    agent: Agent,
    run_dir: Path,
) -> Path:
    """
    Set up the workspace directory structure for an agent run.
    Mimics the Docker container directory structure.

    Returns:
        Path to the workspace directory
    """
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    dirs = {
        "data": workspace / "data",
        "submission": workspace / "submission",
        "logs": workspace / "logs",
        "code": workspace / "code",
        "agent": workspace / "agent",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy public data (read-only for the agent)
    logger.info(f"Setting up workspace data from {competition.public_dir}")
    data_dir = dirs["data"]

    # Create symlinks or copy data
    if competition.public_dir.exists():
        # Use symlinks to save space and time
        for item in competition.public_dir.iterdir():
            link_target = data_dir / item.name
            if not link_target.exists():
                try:
                    link_target.symlink_to(item)
                except OSError:
                    # If symlink fails, copy instead
                    if item.is_file():
                        shutil.copy2(item, link_target)
                    else:
                        shutil.copytree(item, link_target)

    # Copy agent files
    agent_dir = Path(__file__).parent / "agents" / agent.id
    if not agent_dir.exists():
        # Fall back to the directory containing the start script (handles ids like aide/dev)
        agent_dir = agent.start.parent

    if agent_dir.exists():
        for item in agent_dir.iterdir():
            if item.name not in ["Dockerfile", "__pycache__"]:
                dest = dirs["agent"] / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir() and item.name != "__pycache__":
                    shutil.copytree(item, dest, dirs_exist_ok=True)

    # Copy instructions
    instructions_file = Path(__file__).parent / "environment" / "instructions.txt"
    if instructions_file.exists():
        shutil.copy2(instructions_file, workspace / "instructions.txt")

    # Copy validation script
    validation_script = Path(__file__).parent / "environment" / "validate_submission.sh"
    if validation_script.exists():
        shutil.copy2(validation_script, workspace / "validate_submission.sh")
        # Make it executable
        os.chmod(workspace / "validate_submission.sh", 0o755)

    return workspace


async def run_agent_in_conda_env(
    workspace: Path,
    agent: Agent,
    competition: Competition,
    run_logger: logging.Logger,
) -> bool:
    """
    Execute the agent's start script in the conda environment.

    Returns:
        True if successful, False otherwise
    """
    agent_dir = workspace / "agent"
    start_script = agent_dir / Path(agent.start).name

    if not start_script.exists():
        run_logger.error(f"Start script not found: {start_script}")
        return False

    # Set up environment variables
    env = os.environ.copy()
    env.update({
        "SUBMISSION_DIR": str(workspace / "submission"),
        "LOGS_DIR": str(workspace / "logs"),
        "CODE_DIR": str(workspace / "code"),
        "AGENT_DIR": str(agent_dir),
        "COMPETITION_ID": competition.id,
        # Docker-compatible paths for agents expecting /home/* structure
        "DATA_DIR": str(workspace / "data"),
        "INSTRUCTIONS_FILE": str(workspace / "instructions.txt"),
        "VALIDATION_SCRIPT": str(workspace / "validate_submission.sh"),
        "WORKSPACE_DIR": str(workspace),
    })

    # Add agent-specific environment variables
    env.update({k: str(v) for k, v in agent.env_vars.items()})

    # Build the command to run in conda environment
    # We need to activate the conda environment and run the start script
    conda_activate_cmd = f"eval \"$(conda shell.bash hook)\" && conda activate rexagent"
    start_cmd = f"bash {start_script}"

    # Handle agent kwargs
    if agent.kwargs_type == "argparse":
        for key, value in agent.kwargs.items():
            start_cmd += f" --{key} {value}"
    elif agent.kwargs_type == "omegaconf":
        for key, value in agent.kwargs.items():
            start_cmd += f" {key}={value}"

    full_cmd = f"{conda_activate_cmd} && {start_cmd}"

    run_logger.info(f"Executing agent in conda environment 'rexagent'...")
    run_logger.info(f"Working directory: {workspace}")

    try:
        # Run the agent using asyncio subprocess for proper async concurrency
        process = await asyncio.create_subprocess_shell(
            full_cmd,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            executable="/bin/bash",
        )

        assert process.stdout is not None
        buffer = b""
        max_buffer = 128 * 1024
        while True:
            chunk = await process.stdout.read(4096)
            if not chunk:
                if buffer:
                    run_logger.info(f"[Agent] {buffer.decode(errors='replace').rstrip()}")
                    buffer = b""
                break

            buffer += chunk

            while True:
                newline_index = buffer.find(b"\n")
                if newline_index == -1:
                    break
                line = buffer[:newline_index]
                buffer = buffer[newline_index + 1 :]
                run_logger.info(f"[Agent] {line.decode(errors='replace').rstrip()}")

            if len(buffer) > max_buffer:
                run_logger.info(f"[Agent] {buffer.decode(errors='replace').rstrip()}")
                buffer = b""

        # Wait for completion
        returncode = await process.wait()

        if returncode == 0:
            run_logger.info("Agent completed successfully")
            return True
        else:
            run_logger.error(f"Agent exited with code {returncode}")
            return False

    except Exception as e:
        run_logger.error(f"Error running agent: {e}", exc_info=True)
        return False


async def worker(
    idx: int,
    queue: asyncio.Queue[Task],
    tasks_outputs: dict[str, dict[str, Any]],
) -> None:
    """Worker that processes tasks from the queue."""
    while True:
        task = await queue.get()

        # Create logger for the run
        run_logger = get_logger(str(task.path_to_run))
        log_file_handler = logging.FileHandler(task.path_to_run / "run.log")
        log_file_handler.setFormatter(
            logging.getLogger().handlers[0].formatter
        )
        run_logger.addHandler(log_file_handler)
        run_logger.propagate = False

        run_logger.info(
            f"[Worker {idx}] Running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
        )

        task_output = {}
        grading_server = None

        try:
            # Set up workspace
            workspace = setup_agent_workspace(
                task.competition,
                task.agent,
                task.path_to_run,
            )

            # Start grading server with dynamic port allocation
            run_logger.info("Starting grading server...")
            grading_server = GradingServerManager(
                task.competition.id,
                task.competition.private_dir,
                port=None  # Auto-select free port
            )
            grading_server.start()

            # Update environment variable with actual port
            grading_port_file = workspace / ".grading_port"
            grading_port_file.write_text(str(grading_server.port))
            run_logger.info(f"Grading server running on port {grading_server.port}")

            # Wait for server to be ready
            await asyncio.sleep(3)

            # Run agent
            # Using asyncio subprocess to allow parallel execution without
            # fork-in-thread deadlock issues in AIDE's multiprocessing.Process
            success = await run_agent_in_conda_env(
                workspace,
                task.agent,
                task.competition,
                run_logger,
            )

            # Copy outputs to run directory
            for dir_name in ["submission", "logs", "code"]:
                src = workspace / dir_name
                dst = task.path_to_run / dir_name
                if src.exists():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)

            task_output["success"] = success
            run_logger.info(
                f"[Worker {idx}] Finished running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
            )

        except Exception as e:
            stack_trace = traceback.format_exc()
            run_logger.error(f"Error type: {type(e)}")
            run_logger.error(stack_trace)
            run_logger.error(
                f"Run failed for seed {task.seed}, agent {task.agent.id} and competition {task.competition.id}"
            )
            task_output["success"] = False

        finally:
            # Stop grading server
            if grading_server:
                grading_server.stop()

            tasks_outputs[task.run_id] = task_output
            queue.task_done()


async def main(args):
    """Main execution function."""
    global registry
    registry = registry.set_data_dir(Path(args.data_dir))

    agent = agent_registry.get_agent(args.agent_id, config_file=args.config_file)

    run_group = f"{get_timestamp()}_run-group_{agent.name}_no_docker"

    # Load competition ids and check all are prepared
    with open(args.competition_set, "r") as f:
        competition_ids = [line.strip() for line in f.read().splitlines() if line.strip()]

    for competition_id in competition_ids:
        competition = registry.get_challenge(competition_id)
        if not is_dataset_prepared(competition):
            raise ValueError(
                f"Dataset for competition `{competition.id}` is not prepared! "
                f"Please run `rexmle prepare -c {competition.id}` to prepare the dataset."
            )

    # Create tasks for each (competition * seed)
    logger.info(f"Launching run group: {run_group}")
    tasks = []
    for seed in range(args.n_seeds):
        for competition_id in competition_ids:
            competition = registry.get_challenge(competition_id)
            run_dir = create_run_dir(competition.id, agent.id, run_group)
            run_id = run_dir.stem
            task = Task(
                run_id=run_id,
                seed=seed,
                agent=agent,
                competition=competition,
                path_to_run_group=run_dir.parent,
                path_to_run=run_dir,
            )
            tasks.append(task)

    logger.info(f"Creating {args.n_workers} workers to serve {len(tasks)} tasks...")

    # Create queue of tasks and assign workers to run them
    queue = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)

    workers = []
    tasks_outputs = {}
    for idx in range(args.n_workers):
        w = asyncio.create_task(worker(idx, queue, tasks_outputs))
        workers.append(w)

    # Wait for all tasks to be completed and collect results
    started_at = time.monotonic()
    await queue.join()
    time_taken = time.monotonic() - started_at

    for w in workers:
        w.cancel()

    await asyncio.gather(*workers, return_exceptions=True)

    # Generate metadata.json
    metadata = {
        "run_group": run_group,
        "created_at": get_timestamp(),
        "runs": tasks_outputs,
        "no_docker": True,
    }
    run_group_dir = get_runs_dir() / run_group
    with open(run_group_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=False, default=str)

    logger.info(f"{args.n_workers} workers ran for {time_taken:.2f} seconds in total")
    logger.info(f"Results saved to: {run_group_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an agent on a set of competitions WITHOUT Docker."
    )
    parser.add_argument(
        "--agent-id",
        help="Agent ID of the agent to run.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--competition-set",
        type=str,
        required=True,
        help="Path to a text file with a single competition ID on each line",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to run in parallel",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        required=False,
        default=1,
        help="Number of seeds to run for each competition",
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the directory containing the competition data.",
        type=str,
        required=False,
        default=registry.get_data_dir(),
    )
    parser.add_argument(
        "--config-file",
        help="Name of the config file to load (default: config.yaml)",
        type=str,
        required=False,
        default="config.yaml",
    )
    args = parser.parse_args()

    # Check conda is available
    result = subprocess.run(
        ["conda", "--version"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        logger.error("conda is not available. Please install Miniconda or Anaconda.")
        sys.exit(1)

    logger = get_logger(__name__)
    asyncio.run(main(args))
