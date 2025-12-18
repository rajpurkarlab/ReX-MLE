"""
Grading script for PUMA Track 2 Task 1 - Semantic Tissue Segmentation

NOTE: Track 2 Task 1 is IDENTICAL to Track 1 Task 1 (same data, same evaluation)

Evaluation Metric:
- Micro Dice Score: Concatenate all segmentation results and compute Dice score per tissue class,
  then average across classes
- Macro Dice Score: Compute average Dice score per case, then average across all cases
- Background (class 0) is excluded from metric calculation

Class Mapping:
0: tissue_white_background (excluded from evaluation)
1: tissue_stroma
2: tissue_blood_vessel
3: tissue_tumor
4: tissue_epidermis
5: tissue_necrosis

Based on official PUMA evaluation code from:
https://github.com/PUMA-Challenge/PUMA-challenge-eval-track2 (same as track1)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image

from rexmle.grade_helpers import InvalidSubmissionError

# Metric directions for position calculation
METRIC_DIRECTIONS = {
    "dice": "higher_better"
}


def calculate_positions_and_mean(submission_metrics: Dict[str, float], leaderboard_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate positions for each metric and mean position for a submission.

    Args:
        submission_metrics: Dictionary with metric values for the submission
        leaderboard_df: DataFrame containing the leaderboard with existing teams

    Returns:
        Dictionary with position for each metric and mean_position
    """
    # Create a temporary dataframe with submission added
    temp_df = pd.concat([leaderboard_df, pd.DataFrame([submission_metrics])], ignore_index=True)

    positions = {}
    metric_columns = [col for col in temp_df.columns if col not in ['rank', 'team', 'mean_position'] and not col.endswith('_position')]

    for metric in metric_columns:
        if metric in METRIC_DIRECTIONS:
            direction = METRIC_DIRECTIONS[metric]
            if direction == 'lower_better':
                temp_df[f'{metric}_position'] = temp_df[metric].rank(ascending=True, method='min')
            else:  # higher_better
                temp_df[f'{metric}_position'] = temp_df[metric].rank(ascending=False, method='min')

            # Get position for the last row (submission)
            positions[f'{metric}_position'] = temp_df[f'{metric}_position'].iloc[-1]

    # Calculate mean position
    if positions:
        positions['mean_position'] = sum(positions.values()) / len(positions)

    return positions


def calculate_dice_from_masks(mask1: np.ndarray, mask2: np.ndarray, eps: float = 0.00001) -> float:
    """Calculate the DICE score between two binary masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1) + np.sum(mask2)
    dice_score = (2 * intersection + eps) / (union + eps)
    return dice_score


def calculate_dice_per_class(tif1: np.ndarray, tif2: np.ndarray, image_shape: tuple, eps: float = 0.00001) -> Dict[str, float]:
    """
    Calculate the DICE score between two TIF masks per tissue class.

    Args:
        tif1: Ground truth mask
        tif2: Predicted mask
        image_shape: Expected image shape for resizing
        eps: Small epsilon for numerical stability

    Returns:
        Dictionary with Dice scores per tissue class
    """
    # Resize if needed
    if tif1.shape != image_shape:
        tif1 = np.array(Image.fromarray(tif1).resize((image_shape[1], image_shape[0]), Image.NEAREST))
    if tif2.shape != image_shape:
        tif2 = np.array(Image.fromarray(tif2).resize((image_shape[1], image_shape[0]), Image.NEAREST))

    dice_scores = {}
    class_map = {
        1: 'tissue_stroma',
        2: 'tissue_blood_vessel',
        3: 'tissue_tumor',
        4: 'tissue_epidermis',
        5: 'tissue_necrosis'
    }

    for category in range(1, 6):
        # Generate binary masks for each class
        mask1 = np.where(tif1 == category, 1, 0)
        mask2 = np.where(tif2 == category, 1, 0)

        # If both masks are empty, score should be 0 (not informative)
        # Only cases with actual tissue should contribute to the score
        if np.sum(mask1) == 0 and np.sum(mask2) == 0:
            dice_score = 0.0
        else:
            dice_score = calculate_dice_from_masks(mask1, mask2, eps)

        dice_scores[class_map[category]] = dice_score

    # Calculate average dice for this case
    class_scores = [score for score in dice_scores.values()]
    dice_scores['average_dice'] = sum(class_scores) / len(class_scores) if class_scores else 0.0

    return dice_scores


def calculate_micro_dice_score(all_gt_masks: List[np.ndarray], all_pred_masks: List[np.ndarray],
                               image_shape: tuple, eps: float = 0.00001) -> Dict[str, float]:
    """
    Calculate the overall micro DICE score across all classes by concatenating all masks.

    This is the official PUMA Track 1 Task 1 evaluation metric.
    """
    class_map = {
        1: 'tissue_stroma',
        2: 'tissue_blood_vessel',
        3: 'tissue_tumor',
        4: 'tissue_epidermis',
        5: 'tissue_necrosis'
    }

    total_gt_mask = {class_name: [] for class_name in class_map.values()}
    total_pred_mask = {class_name: [] for class_name in class_map.values()}

    # Accumulate masks for each class
    for gt_mask, pred_mask in zip(all_gt_masks, all_pred_masks):
        # Resize if needed
        if gt_mask.shape != image_shape:
            gt_mask = np.array(Image.fromarray(gt_mask).resize((image_shape[1], image_shape[0]), Image.NEAREST))
        if pred_mask.shape != image_shape:
            pred_mask = np.array(Image.fromarray(pred_mask).resize((image_shape[1], image_shape[0]), Image.NEAREST))

        for category, class_name in class_map.items():
            gt_binary = np.where(gt_mask == category, 1, 0)
            pred_binary = np.where(pred_mask == category, 1, 0)

            total_gt_mask[class_name].append(gt_binary)
            total_pred_mask[class_name].append(pred_binary)

    # Concatenate all masks for each class along height axis
    for class_name in class_map.values():
        total_gt_mask[class_name] = np.concatenate(total_gt_mask[class_name], axis=0)
        total_pred_mask[class_name] = np.concatenate(total_pred_mask[class_name], axis=0)

    # Calculate the micro DICE score for each class
    micro_dice_scores = {}
    for class_name in class_map.values():
        mask1 = total_gt_mask[class_name]
        mask2 = total_pred_mask[class_name]

        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1) + np.sum(mask2)

        # Calculate dice score
        # If both masks are empty (union == 0), score is 0 (not informative)
        # Only cases with actual tissue should contribute meaningfully to the metric
        if union == 0:
            dice_score = 0.0  # Both masks empty = no information
        elif intersection == 0:
            dice_score = 0.0  # No overlap = no match
        else:
            dice_score = (2 * intersection + eps) / (union + eps)

        micro_dice_scores[class_name] = dice_score

    # Average micro dice across all tissue classes
    average_dice_score = np.mean(list(micro_dice_scores.values()))
    micro_dice_scores['average_micro_dice'] = average_dice_score

    return micro_dice_scores


def grade(submission: pd.DataFrame, answers: pd.DataFrame, submission_dir: Optional[Path] = None, answers_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Grade a PUMA Track 1 Task 1 submission (Tissue Segmentation).

    Evaluation follows the official PUMA Challenge methodology:
    - Compute per-case Dice scores for each tissue class
    - Compute micro Dice score by concatenating all masks
    - Report macro Dice (average per-case Dice) and micro Dice

    Args:
        submission: Submission DataFrame with columns: case_id, predicted_mask_path
        answers: Ground truth DataFrame with columns: case_id, image_path, label_path
        submission_dir: Directory containing submission files
        answers_dir: Directory containing ground truth files

    Returns:
        Dictionary with micro_dice (primary metric) and macro_dice
    """
    if submission_dir is None:
        raise InvalidSubmissionError("Segmentation task requires submission_dir")

    if answers_dir is None:
        raise InvalidSubmissionError("Segmentation task requires answers_dir")

    # Validate submission format
    if 'case_id' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'case_id' column")

    if 'predicted_mask_path' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'predicted_mask_path' column")

    # Check answers format
    assert 'case_id' in answers.columns, "Answers must contain 'case_id' column"
    assert 'label_path' in answers.columns, "Answers must contain 'label_path' column"

    # Check IDs match
    submission_ids = set(submission['case_id'])
    answer_ids = set(answers['case_id'])

    if submission_ids != answer_ids:
        missing = answer_ids - submission_ids
        extra = submission_ids - answer_ids

        error_msg = "Submission IDs don't match ground truth"
        if missing:
            error_msg += f"\n  Missing IDs: {list(missing)[:5]}..."
        if extra:
            error_msg += f"\n  Extra IDs: {list(extra)[:5]}..."

        raise InvalidSubmissionError(error_msg)

    # Expected image shape (1024x1024 as per PUMA specs)
    image_shape = (1024, 1024)

    # Collect all masks and per-case metrics
    all_pred_masks = []
    all_gt_masks = []
    per_case_dice_scores = []

    for idx, row in submission.iterrows():
        case_id = row['case_id']
        pred_mask_path = submission_dir / row['predicted_mask_path']

        # Find ground truth
        gt_row = answers[answers['case_id'] == case_id]
        if gt_row.empty:
            raise InvalidSubmissionError(f"No ground truth found for {case_id}")

        gt_mask_path = answers_dir / gt_row.iloc[0]['label_path']

        # Validate predicted mask exists
        if not pred_mask_path.exists():
            raise InvalidSubmissionError(f"Predicted mask not found: {pred_mask_path}")

        # Load masks (TIF or PNG format)
        try:
            pred_mask = np.array(Image.open(pred_mask_path))
        except Exception as e:
            raise InvalidSubmissionError(f"Error loading predicted mask {pred_mask_path}: {e}")

        try:
            # Ground truth can be GeoJSON (need conversion) or already converted mask
            gt_mask = np.array(Image.open(gt_mask_path))
        except Exception as e:
            raise InvalidSubmissionError(f"Error loading ground truth mask {gt_mask_path}: {e}")

        # Validate prediction is multi-class with correct range (0-5)
        pred_unique = np.unique(pred_mask)
        if pred_unique.max() > 5:
            raise InvalidSubmissionError(
                f"Prediction mask for {case_id} contains invalid class IDs. "
                f"Max class ID: {pred_unique.max()}, expected max: 5"
            )

        # Calculate per-case Dice scores
        case_dice = calculate_dice_per_class(gt_mask, pred_mask, image_shape)
        per_case_dice_scores.append(case_dice['average_dice'])

        # Store masks for micro Dice calculation
        all_pred_masks.append(pred_mask)
        all_gt_masks.append(gt_mask)

    # Compute micro Dice score (official PUMA metric)
    micro_metrics = calculate_micro_dice_score(all_gt_masks, all_pred_masks, image_shape)

    # Compute macro Dice score (average of per-case Dice scores)
    macro_dice = np.mean(per_case_dice_scores)

    # Print summary
    print(f"\n=== PUMA Track 2 Task 1 (Tissue Segmentation) Results ===")
    print(f"Cases evaluated: {len(all_pred_masks)}")
    print(f"Micro Dice Score (Primary): {micro_metrics['average_micro_dice']:.4f}")
    print(f"  - Stroma: {micro_metrics['tissue_stroma']:.4f}")
    print(f"  - Blood Vessel: {micro_metrics['tissue_blood_vessel']:.4f}")
    print(f"  - Tumor: {micro_metrics['tissue_tumor']:.4f}")
    print(f"  - Epidermis: {micro_metrics['tissue_epidermis']:.4f}")
    print(f"  - Necrosis: {micro_metrics['tissue_necrosis']:.4f}")
    # print(f"Macro Dice Score: {macro_dice:.4f}")

    # Build results dictionary
    results_dict = {
        "dice": float(micro_metrics['average_micro_dice']),
    }

    # Calculate position and mean position if leaderboard exists
    mean_position_value = None
    leaderboard_path = Path(__file__).parent / 'leaderboard.csv'

    if leaderboard_path.exists():
        try:
            leaderboard_df = pd.read_csv(leaderboard_path)
            submission_dict = {"dice": results_dict["dice"]}
            positions = calculate_positions_and_mean(submission_dict, leaderboard_df)
            mean_position_value = positions.get('mean_position')
            # Add individual metric positions to results

            for metric_name, position_value in positions.items():

                if metric_name != 'mean_position':

                    results_dict[metric_name] = position_value


            results_dict['mean_position'] = mean_position_value

            # Calculate percentile: 1 - ((mean_position - 1) / num_competitors)
            num_competitors = len(leaderboard_df)
            percentile = 1 - ((mean_position_value - 1) / (num_competitors))
            results_dict['percentile'] = percentile

            print(f"Mean Position: {mean_position_value:.2f}")
            print(f"Percentile: {percentile:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate position: {e}")

    # Set 'overall' for ranking (lower mean_position is better)
    if mean_position_value is not None:
        results_dict['overall'] = mean_position_value
    else:
        # Fallback: use negative dice for ranking (higher dice = lower/better rank)
        results_dict['overall'] = -results_dict['dice']

    return results_dict