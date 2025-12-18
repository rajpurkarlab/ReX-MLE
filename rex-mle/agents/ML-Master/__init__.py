from dataclasses import dataclass

try:
    from .backend import compile_prompt_to_md
except ImportError:
    from backend import compile_prompt_to_md  # type: ignore

try:
    from agent.agent import Agent
    from interpreter.interpreter import Interpreter
    from search.journal import Journal
    from search.node import Node
    from omegaconf import OmegaConf
    from rich.status import Status
    from .utils.config import (
        load_task_desc,
        prep_agent_workspace,
        save_run,
        _load_cfg,
        prep_cfg,
    )
    from pathlib import Path
except ImportError:
    Agent = None
    Interpreter = None
    Journal = None
    Node = None
    OmegaConf = None
    Status = None
    Path = None

@dataclass
class Solution:
    code: str
    valid_metric: float

class Experiment:

    def __init__(self, data_dir: str, goal: str, eval: str | None = None):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """
        if Agent is None or Interpreter is None:
            raise ImportError("Full agent stack is unavailable; ensure ML-Master is installed as a package.")
        
        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        self.cfg = prep_cfg(_cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir, **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )

    def run(self, steps: int) -> Solution:
        for _i in range(steps):
            self.agent.step(exec_callback=self.interpreter.run)
            save_run(self.cfg, self.journal)
        self.interpreter.cleanup_session()

        best_node = self.journal.get_best_node()
        return Solution(code=best_node.code, valid_metric=best_node.metric.value)
