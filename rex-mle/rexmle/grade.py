"""
High-level batch grading functionality for Med-MLE-Bench.

Supports grading multiple challenge submissions from a JSONL file.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
import pandas as pd
import logging

from tqdm import tqdm

from rexmle.registry import Challenge, Registry
from rexmle.registry import registry as DEFAULT_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class ChallengeReport:
    """Report for a single challenge submission."""
    challenge_id: str
    score: Optional[float]
    metrics: Optional[Dict[str, float]]  # All metrics if multi-metric
    gold_threshold: Optional[float]
    silver_threshold: Optional[float]
    bronze_threshold: Optional[float]
    median_threshold: Optional[float]
    any_medal: bool
    gold_medal: bool
    silver_medal: bool
    bronze_medal: bool
    above_median: bool
    submission_exists: bool
    valid_submission: bool
    is_lower_better: bool
    created_at: str
    submission_path: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        import numpy as np

        d = asdict(self)
        # Convert datetime to string if it's a datetime object
        if isinstance(d['created_at'], datetime):
            d['created_at'] = d['created_at'].isoformat()

        # Convert numpy/pandas types to native Python types for JSON serialization
        for key, value in d.items():
            if isinstance(value, (np.bool_, np.integer)):
                d[key] = bool(value) if isinstance(value, np.bool_) else int(value)
            elif isinstance(value, np.floating):
                d[key] = float(value)
            elif isinstance(value, dict):
                # Recursively convert nested dict values
                for k, v in value.items():
                    if isinstance(v, (np.bool_, np.integer)):
                        value[k] = bool(v) if isinstance(v, np.bool_) else int(v)
                    elif isinstance(v, np.floating):
                        value[k] = float(v)

        return d


def grade_jsonl(
    path_to_submissions: Path,
    output_dir: Path,
    registry: Registry = DEFAULT_REGISTRY,
    suffix: str = '',
):
    """
    Grades multiple submissions stored in a JSONL file.
    Saves the aggregated report as a JSON file.

    JSONL format (one submission per line):
    {"challenge_id": "topcow-track1-task1", "submission_path": "/path/to/submission.csv", "submission_dir": "/path/to/masks"}

    Note: submission_dir is optional and only needed for segmentation/detection tasks.

    Args:
        path_to_submissions: Path to JSONL file containing submissions
        output_dir: Directory where grading report will be saved
        registry: Challenge registry to use
        suffix: Optional text to append to the filename (before .json)
    """
    # Read submissions
    try:
        with open(path_to_submissions, 'r') as f:
            submissions = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                try:
                    submissions.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: Invalid JSON - {e}")
    except FileNotFoundError:
        logger.error(f"Submission file not found: {path_to_submissions}")
        raise

    if not submissions:
        logger.error("No valid submissions found in JSONL file")
        raise ValueError("No valid submissions found")

    logger.info(f"Found {len(submissions)} submission(s) to grade")

    challenge_reports = []

    for submission in tqdm(submissions, desc="Grading submissions", unit="submission"):
        submission_path = Path(str(submission["submission_path"]))
        challenge_id = submission["challenge_id"]
        submission_dir = submission.get("submission_dir")
        print(f'grading challenge {challenge_id} with submission {submission_path}')
        if submission_dir:
            submission_dir = Path(submission_dir)

        try:
            challenge = registry.get_challenge(challenge_id)
            single_report = grade_csv(submission_path, challenge, submission_dir=submission_dir)
            challenge_reports.append(single_report)
        except Exception as e:
            logger.error(f"Error grading {challenge_id}: {e}")
            # Create a failed report
            challenge_reports.append(ChallengeReport(
                challenge_id=challenge_id,
                score=None,
                metrics=None,
                gold_threshold=None,
                silver_threshold=None,
                bronze_threshold=None,
                median_threshold=None,
                any_medal=False,
                gold_medal=False,
                silver_medal=False,
                bronze_medal=False,
                above_median=False,
                submission_exists=submission_path.exists(),
                valid_submission=False,
                is_lower_better=False,
                created_at=datetime.now().isoformat(),
                submission_path=str(submission_path),
            ))

    # Aggregate reports
    aggregated_report = aggregate_reports(challenge_reports)

    # Save report
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-GMT")
    suffix_str = f"_{suffix}" if suffix else ""
    save_path = output_dir / f"{timestamp}{suffix_str}_grading_report.json"

    # Print summary
    summary = {k: v for k, v in aggregated_report.items() if k != "challenge_reports"}
    logger.info("\n" + "="*70)
    logger.info("GRADING SUMMARY")
    logger.info("="*70)
    logger.info(json.dumps(summary, indent=2))
    logger.info("="*70)

    # Save full report
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(aggregated_report, f, indent=2)

    logger.info(f"\n✓ Saved full report to: {save_path}\n")

    return aggregated_report


def grade_csv(
    path_to_submission: Path,
    challenge: Challenge,
    submission_dir: Optional[Path] = None
) -> ChallengeReport:
    """
    Grades a submission CSV for the given challenge.

    Args:
        path_to_submission: Path to submission CSV file
        challenge: Challenge object
        submission_dir: Optional directory containing submission files (e.g., masks)

    Returns:
        ChallengeReport with grading results
    """
    # Check if dataset is prepared
    if not challenge.answers.exists():
        raise ValueError(
            f"Dataset for challenge `{challenge.id}` is not prepared! "
            f"Ground truth not found at: {challenge.answers}"
        )

    score = None
    metrics = None
    submission_exists = path_to_submission.is_file() and path_to_submission.suffix.lower() == ".csv"

    if submission_exists:
        try:
            submission_df = pd.read_csv(path_to_submission)
            answers_df = pd.read_csv(challenge.answers)

            # Determine submission_dir if not provided
            if submission_dir is None:
                submission_dir = path_to_submission.parent

            # Grade submission
            result = challenge.grader(
                submission_df,
                answers_df,
                submission_dir=submission_dir,
                answers_dir=challenge.answers_dir
            )

            # Handle dict vs float return
            if isinstance(result, dict):
                metrics = result
                # Use 'overall' metric for ranking if available, otherwise first metric
                if 'overall' in result:
                    score = result['overall']
                else:
                    score = next(iter(result.values()))
            else:
                score = result

        except Exception as e:
            logger.error(f"Error grading submission for {challenge.id}: {e}")
            score = None
            metrics = None
    else:
        logger.warning(
            f"Invalid submission file: {path_to_submission}. "
            f"Please check that the file exists and it is a CSV."
        )

    valid_submission = score is not None

    # Get leaderboard info if available
    if challenge.leaderboard and challenge.leaderboard.exists():
        try:
            leaderboard = pd.read_csv(challenge.leaderboard)

            # Check if leaderboard has mean_position column (new ranking system)
            if 'mean_position' in leaderboard.columns:
                # Use mean_position values from ranks 1, 2, 3 as thresholds
                is_lower_better = True  # Positions are always lower-is-better
                
                # Extract mean_position from ranks 1, 2, 3
                # Ensure leaderboard is sorted by rank
                if 'rank' in leaderboard.columns:
                    leaderboard_sorted = leaderboard.sort_values('rank')
                else:
                    # If no rank column, assume first row is rank 1
                    leaderboard_sorted = leaderboard
                
                mean_positions = leaderboard_sorted['mean_position'].values
                
                # Get thresholds from ranks 1, 2, 3 (convert to float for JSON serialization)
                gold_threshold = float(mean_positions[0]) if len(mean_positions) > 0 else None
                silver_threshold = float(mean_positions[1]) if len(mean_positions) > 1 else None
                bronze_threshold = float(mean_positions[2]) if len(mean_positions) > 2 else None
                
                # Median threshold: use median of mean_position values
                median_threshold = float(mean_positions[len(mean_positions) // 2]) if len(mean_positions) > 0 else None
                
                # Use submission's mean_position for comparison (stored as 'overall' in metrics)
                # When leaderboard has mean_position, the submission's mean_position is in metrics['overall']
                comparison_score = None
                if metrics and isinstance(metrics, dict) and 'overall' in metrics:
                    comparison_score = metrics['overall']
                elif score is not None:
                    comparison_score = score
                
                # Detect if this is a single-metric leaderboard (mean_positions are whole numbers)
                # When there's only 1 metric, mean_position equals the position for that metric (1.0, 2.0, 3.0, etc.)
                metric_cols = [col for col in leaderboard.columns
                              if not col.endswith('_position') and col not in ['rank', 'team', 'mean_position']]
                is_single_metric = len(metric_cols) == 1
                
                # Determine medals based on mean_position comparison
                if comparison_score is None:
                    gold_medal = False
                    silver_medal = False
                    bronze_medal = False
                    above_median = False
                else:
                    # For mean_position system, lower is always better
                    # Use <= (less than or equal) to handle whole number mean_positions correctly
                    # When there's only 1 metric, mean_positions are whole numbers (1.0, 2.0, 3.0, etc.)
                    # so a submission with mean_position 1.0 should get gold if threshold is 1.0
                    gold_medal = gold_threshold is not None and comparison_score <= gold_threshold
                    silver_medal = (silver_threshold is not None and comparison_score <= silver_threshold and 
                                   not gold_medal)
                    bronze_medal = (bronze_threshold is not None and comparison_score <= bronze_threshold and 
                                   not silver_medal and not gold_medal)
                    above_median = (median_threshold is not None and comparison_score <= median_threshold)
                
                rank_info = {
                    "gold_medal": gold_medal,
                    "silver_medal": silver_medal,
                    "bronze_medal": bronze_medal,
                    "above_median": above_median,
                    "gold_threshold": gold_threshold,
                    "silver_threshold": silver_threshold,
                    "bronze_threshold": bronze_threshold,
                    "median_threshold": median_threshold,
                }
            else:
                # Fall back to old system using 'score' column
                rank_info = challenge.grader.rank_score(score, leaderboard)
                is_lower_better = challenge.grader.is_lower_better(leaderboard)

        except Exception as e:
            logger.warning(f"Error reading leaderboard for {challenge.id}: {e}")
            rank_info = {
                "gold_medal": False,
                "silver_medal": False,
                "bronze_medal": False,
                "above_median": False,
                "gold_threshold": None,
                "silver_threshold": None,
                "bronze_threshold": None,
                "median_threshold": None,
            }
            is_lower_better = False
    else:
        rank_info = {
            "gold_medal": False,
            "silver_medal": False,
            "bronze_medal": False,
            "above_median": False,
            "gold_threshold": None,
            "silver_threshold": None,
            "bronze_threshold": None,
            "median_threshold": None,
        }
        is_lower_better = False

    return ChallengeReport(
        challenge_id=challenge.id,
        score=score,
        metrics=metrics,
        gold_threshold=rank_info["gold_threshold"],
        silver_threshold=rank_info["silver_threshold"],
        bronze_threshold=rank_info["bronze_threshold"],
        median_threshold=rank_info["median_threshold"],
        any_medal=(rank_info["gold_medal"] or rank_info["silver_medal"] or rank_info["bronze_medal"]),
        gold_medal=rank_info["gold_medal"],
        silver_medal=rank_info["silver_medal"],
        bronze_medal=rank_info["bronze_medal"],
        above_median=rank_info["above_median"],
        submission_exists=submission_exists,
        valid_submission=valid_submission,
        is_lower_better=is_lower_better,
        created_at=datetime.now().isoformat(),
        submission_path=str(path_to_submission),
    )

def validate_submission(submission: Path, challenge: Challenge) -> tuple[bool, str]:
    """
    Validates a submission for the given challenge by actually running the challenge grader.
    This is designed for end users, not developers (we assume that the challenge grader is
    correctly implemented and use that for validating the submission, not the other way around).
    """
    if not submission.is_file():
        return False, f"Submission invalid! Submission file {submission} does not exist."

    if not submission.suffix.lower() == ".csv":
        return False, "Submission invalid! Submission file must be a CSV file."

    # Check if answers file exists (simpler than is_dataset_prepared)
    if not challenge.answers.exists():
        raise ValueError(
            f"Dataset for challenge `{challenge.id}` is not prepared! "
            f"Ground truth not found at: {challenge.answers}"
        )

    try:
        # Read submission and answers
        submission_df = pd.read_csv(submission)
        answers_df = pd.read_csv(challenge.answers)

        # Determine submission_dir
        submission_dir = submission.parent

        # Run grader to validate
        challenge.grader(
            submission_df,
            answers_df,
            submission_dir=submission_dir,
            answers_dir=challenge.answers_dir
        )
    except Exception as e:
        return (
            False,
            f"Submission invalid! The attempt to grade the submission has resulted in the following error message:\n{e}",
        )

    return True, "Submission is valid."

def aggregate_reports(challenge_reports: List[ChallengeReport]) -> dict:
    """
    Builds the summary report from a list of challenge reports.

    Args:
        challenge_reports: List of ChallengeReport objects

    Returns:
        Dictionary with aggregated statistics and individual reports
    """
    total_gold_medals = sum(report.gold_medal for report in challenge_reports)
    total_silver_medals = sum(report.silver_medal for report in challenge_reports)
    total_bronze_medals = sum(report.bronze_medal for report in challenge_reports)
    total_above_median = sum(report.above_median for report in challenge_reports)
    total_submissions = sum(report.submission_exists for report in challenge_reports)
    total_valid_submissions = sum(report.valid_submission for report in challenge_reports)

    summary_report = {
        "total_runs": int(len(challenge_reports)),
        "total_runs_with_submissions": int(total_submissions),
        "total_valid_submissions": int(total_valid_submissions),
        "total_medals": int(total_gold_medals + total_silver_medals + total_bronze_medals),
        "total_gold_medals": int(total_gold_medals),
        "total_silver_medals": int(total_silver_medals),
        "total_bronze_medals": int(total_bronze_medals),
        "total_above_median": int(total_above_median),
        "challenge_reports": [cr.to_dict() for cr in challenge_reports],
    }

    return summary_report
