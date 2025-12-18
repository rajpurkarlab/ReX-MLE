"""
Helper classes for grading submissions.

Supports multiple submission formats:
- Classification/Regression: CSV file with predictions
- Segmentation: CSV file + directory with mask files
- Detection: CSV file with bounding boxes
"""

import inspect
from dataclasses import dataclass
from typing import Any, Optional, Callable, Union, Dict
from pathlib import Path
import pandas as pd
import logging

from rexmle.utils import import_fn

logger = logging.getLogger(__name__)


class InvalidSubmissionError(Exception):
    """Exception raised when a submission is invalid."""
    pass


class Grader:
    """
    Grader for evaluating challenge submissions.
    """

    def __init__(self, name: str, grade_fn: str) -> None:
        """
        Initialize grader.

        Args:
            name: Name of grading metric (e.g., "dice", "auc", "accuracy")
            grade_fn: Module path to grading function (e.g., "rexmle.challenges.challenge1.grade:grade")
        """
        self.name = name
        self.grade_fn = import_fn(grade_fn)
        assert isinstance(self.name, str), "Grader name must be a string."
        assert len(self.name) > 0, "Grader name cannot be empty."

    def is_lower_better(self, leaderboard: pd.DataFrame) -> bool:
        """
        Determines if a lower score is better based on the leaderboard.

        Args:
            leaderboard: Leaderboard DataFrame with 'score' column

        Returns:
            True if lower scores are better, False otherwise
        """
        scores = leaderboard["score"]
        top_score = scores.iloc[0]
        bottom_score = scores.iloc[-1]

        return bool(top_score < bottom_score)

    @staticmethod
    def from_dict(data: dict) -> "Grader":
        """
        Create Grader from dictionary.

        Args:
            data: Dictionary with 'name' and 'grade_fn' keys

        Returns:
            Grader instance
        """
        return Grader(**data)

    def __call__(self, submission: Union[pd.DataFrame, Path], answers: Any, submission_dir: Optional[Path] = None, answers_dir: Optional[Path] = None) -> Optional[Union[float, Dict[str, float]]]:
        """
        Runs the grading function on a submission.

        Args:
            submission: Submission DataFrame or Path to submission CSV
            answers: Ground truth answers
            submission_dir: Optional directory containing submission files (e.g., segmentation masks)
            answers_dir: Optional directory containing ground truth files (e.g., ground truth masks)

        Returns:
            Dictionary of metrics with values, or a single score (for backward compatibility), or None if invalid/error
            - If grade function returns a dict: returns dict with all metrics rounded to 5 decimal places
            - If grade function returns a float: returns float rounded to 5 decimal places
        """
        # Load submission if it's a path
        if isinstance(submission, (str, Path)):
            submission_path = Path(submission)
            if not submission_path.exists():
                logger.error(f"Submission file not found: {submission_path}")
                return None
            submission = pd.read_csv(submission_path)

            # If submission_dir not provided, use parent directory
            if submission_dir is None:
                submission_dir = submission_path.parent

        try:
            # Call grading function with appropriate arguments
            # Check which parameters the grade_fn accepts
            sig = inspect.signature(self.grade_fn)
            kwargs = {}
            if 'submission_dir' in sig.parameters:
                kwargs['submission_dir'] = submission_dir
            if 'answers_dir' in sig.parameters:
                kwargs['answers_dir'] = answers_dir

            result = self.grade_fn(submission, answers, **kwargs)
        except InvalidSubmissionError as e:
            logger.warning(f"Invalid submission: {e}")
            return None
        except Exception as e:
            try:
                fpath = inspect.getfile(self.grade_fn)
                line_number = inspect.getsourcelines(self.grade_fn)[1]
                fn_info = f"{fpath}:{line_number}"
            except TypeError:
                fn_info = str(self.grade_fn)
            logger.error(f"Unexpected error during grading: {e}. Check {fn_info}")
            return None

        # Handle dictionary return (multiple metrics)
        if isinstance(result, dict):
            # Round all metric values to 5 decimal places
            rounded_result = {k: round(v, 5) for k, v in result.items()}
            return rounded_result
        # Handle single score return (backward compatibility)
        else:
            rounded_score = round(result, 5)
            return rounded_score

    def rank_score(self, score: Optional[float], leaderboard: pd.DataFrame) -> dict:
        """
        Ranks a score based on the leaderboard (following Kaggle medal system).

        Args:
            score: Score to rank
            leaderboard: Leaderboard DataFrame with 'score' column

        Returns:
            Dictionary with medal thresholds and rankings
        """
        assert "score" in leaderboard.columns, "Leaderboard must have a `score` column."

        lower_is_better = self.is_lower_better(leaderboard)

        num_teams = len(leaderboard)
        scores = leaderboard["score"]

        def get_score_at_position(position: int) -> float:
            """Get score at given position (1-indexed)."""
            if position - 1 >= len(scores) or position < 1:
                raise IndexError("Position out of bounds in the leaderboard.")
            return scores.iloc[position - 1]

        def get_thresholds(num_teams: int) -> tuple:
            """Get medal thresholds based on Kaggle progression system."""
            if 1 <= num_teams < 100:
                gold_threshold = get_score_at_position(max(1, int(num_teams * 0.1)))
                silver_threshold = get_score_at_position(max(1, int(num_teams * 0.2)))
                bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.4)))
            elif 100 <= num_teams < 250:
                gold_threshold = get_score_at_position(10)
                silver_threshold = get_score_at_position(max(1, int(num_teams * 0.2)))
                bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.4)))
            elif 250 <= num_teams < 1000:
                gold_threshold = get_score_at_position(10 + int(num_teams * 0.002))
                silver_threshold = get_score_at_position(10 + int(num_teams * 0.05))
                bronze_threshold = get_score_at_position(10 + int(num_teams * 0.1))
            else:  # >= 1000
                gold_threshold = get_score_at_position(10 + int(num_teams * 0.002))
                silver_threshold = get_score_at_position(10 + int(num_teams * 0.05))
                bronze_threshold = get_score_at_position(10 + int(num_teams * 0.1))

            # Median threshold
            median_position = num_teams // 2
            median_threshold = get_score_at_position(median_position)

            return gold_threshold, silver_threshold, bronze_threshold, median_threshold

        gold_threshold, silver_threshold, bronze_threshold, median_threshold = get_thresholds(num_teams)

        # Rank score
        if score is None:
            return {
                "gold_medal": False,
                "silver_medal": False,
                "bronze_medal": False,
                "above_median": False,
                "gold_threshold": gold_threshold,
                "silver_threshold": silver_threshold,
                "bronze_threshold": bronze_threshold,
                "median_threshold": median_threshold,
            }

        if lower_is_better:
            gold_medal = score <= gold_threshold
            silver_medal = score <= silver_threshold and not gold_medal
            bronze_medal = score <= bronze_threshold and not silver_medal and not gold_medal
            above_median = score <= median_threshold
        else:
            gold_medal = score >= gold_threshold
            silver_medal = score >= silver_threshold and not gold_medal
            bronze_medal = score >= bronze_threshold and not silver_medal and not gold_medal
            above_median = score >= median_threshold

        return {
            "gold_medal": gold_medal,
            "silver_medal": silver_medal,
            "bronze_medal": bronze_medal,
            "above_median": above_median,
            "gold_threshold": gold_threshold,
            "silver_threshold": silver_threshold,
            "bronze_threshold": bronze_threshold,
            "median_threshold": median_threshold,
        }
