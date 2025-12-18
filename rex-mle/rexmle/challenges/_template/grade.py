"""
Grading script for [CHALLENGE NAME]

This script evaluates submissions and computes the challenge metric.

For segmentation tasks, the grade function can accept a submission_dir parameter
that contains the segmentation mask files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from rexmle.grade_helpers import InvalidSubmissionError
from rexmle.medical_utils import load_medical_image


def prepare_for_grading(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_column: str = "image_id",
    target_column: str = "prediction"
) -> tuple:
    """
    Validate and prepare submission for grading.

    Args:
        submission: Submission DataFrame
        answers: Ground truth DataFrame
        id_column: Name of ID column
        target_column: Name of prediction column

    Returns:
        Tuple of (y_pred, y_true) arrays

    Raises:
        InvalidSubmissionError: If submission format is invalid
    """
    # Check required columns
    if id_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain column '{id_column}'")

    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain column '{target_column}'")

    # Check answers columns
    assert id_column in answers.columns, f"Answers must contain column '{id_column}'"
    assert "label" in answers.columns, "Answers must contain column 'label'"

    # Check not empty
    if submission.empty:
        raise InvalidSubmissionError("Submission DataFrame is empty")

    assert not answers.empty, "Answers DataFrame is empty"

    # Check IDs match
    submission_ids = set(submission[id_column])
    answer_ids = set(answers[id_column])

    if submission_ids != answer_ids:
        missing = answer_ids - submission_ids
        extra = submission_ids - answer_ids

        error_msg = "Submission IDs don't match ground truth"
        if missing:
            error_msg += f"\n  Missing IDs: {list(missing)[:5]}..."
        if extra:
            error_msg += f"\n  Extra IDs: {list(extra)[:5]}..."

        raise InvalidSubmissionError(error_msg)

    # Check for missing values
    if submission[target_column].isnull().any():
        raise InvalidSubmissionError("Submission contains missing prediction values")

    # Sort both by ID to ensure alignment
    submission_sorted = submission.sort_values(by=id_column).reset_index(drop=True)
    answers_sorted = answers.sort_values(by=id_column).reset_index(drop=True)

    # Extract prediction and true values
    y_pred = submission_sorted[target_column].values
    y_true = answers_sorted["label"].values

    return y_pred, y_true


def grade(submission: pd.DataFrame, answers: pd.DataFrame, submission_dir: Optional[Path] = None) -> float:
    """
    Grade a submission.

    Args:
        submission: Submission DataFrame with predictions
        answers: Ground truth DataFrame with labels
        submission_dir: Optional directory containing submission files (for segmentation tasks)

    Returns:
        Score (float)

    Raises:
        InvalidSubmissionError: If submission format is invalid
    """
    # ===================================================================
    # CUSTOMIZE THIS SECTION FOR YOUR CHALLENGE
    # ===================================================================

    # OPTION 1: Classification/Regression (CSV-only)
    # -----------------------------------------------
    # Prepare for grading
    y_pred, y_true = prepare_for_grading(
        submission=submission,
        answers=answers,
        id_column="image_id",  # Customize for your challenge
        target_column="prediction"  # Customize for your challenge
    )

    # Example: Binary classification with AUC
    score = roc_auc_score(y_true, y_pred)

    # Example: Multi-class classification with accuracy
    # y_pred_class = np.argmax(y_pred, axis=1)  # If predictions are probabilities
    # score = accuracy_score(y_true, y_pred_class)

    # Example: Regression with MSE
    # from sklearn.metrics import mean_squared_error
    # score = mean_squared_error(y_true, y_pred)

    # OPTION 2: Segmentation (CSV + mask files)
    # ------------------------------------------
    # if submission_dir is None:
    #     raise InvalidSubmissionError("Segmentation task requires submission_dir")
    #
    # # Validate CSV has required columns
    # if "image_id" not in submission.columns or "mask_path" not in submission.columns:
    #     raise InvalidSubmissionError("Submission must contain 'image_id' and 'mask_path' columns")
    #
    # # Load ground truth directory (assumes answers CSV has ground_truth_path column)
    # ground_truth_dir = answers.iloc[0].get('ground_truth_dir', submission_dir.parent / 'ground_truth')
    #
    # dice_scores = []
    # for idx, row in submission.iterrows():
    #     image_id = row['image_id']
    #     pred_mask_path = submission_dir / row['mask_path']
    #
    #     # Find corresponding ground truth
    #     gt_row = answers[answers['image_id'] == image_id]
    #     if gt_row.empty:
    #         raise InvalidSubmissionError(f"No ground truth found for {image_id}")
    #
    #     gt_mask_path = Path(gt_row.iloc[0]['mask_path'])
    #
    #     # Load masks
    #     if not pred_mask_path.exists():
    #         raise InvalidSubmissionError(f"Predicted mask not found: {pred_mask_path}")
    #
    #     pred_mask = load_medical_image(pred_mask_path)['image']
    #     gt_mask = load_medical_image(gt_mask_path)['image']
    #
    #     # Compute Dice coefficient
    #     intersection = np.logical_and(pred_mask, gt_mask).sum()
    #     dice = 2.0 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-7)
    #     dice_scores.append(dice)
    #
    # # Average Dice score
    # score = np.mean(dice_scores)

    # OPTION 3: Detection (CSV with bounding boxes)
    # ----------------------------------------------
    # # Expected columns: image_id, x_min, y_min, x_max, y_max, confidence
    # from sklearn.metrics import average_precision_score
    # # Implement IoU matching and compute mAP
    # # score = compute_map(submission, answers)

    # ===================================================================
    # END CUSTOMIZATION
    # ===================================================================

    return score
