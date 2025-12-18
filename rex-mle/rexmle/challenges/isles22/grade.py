"""
Grading script for ISLES'22 - Ischemic Stroke Lesion Segmentation Challenge
Computes Dice, lesion-wise F1, lesion count difference, and absolute volume difference.
"""

__author__ = "Ezequiel de la Rosa"

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import cc3d
import nibabel as nib
import numpy as np
import pandas as pd

from rexmle.grade_helpers import InvalidSubmissionError


# ===========================
# Core Metric Functions
# ===========================

def compute_dice(im1: np.ndarray, im2: np.ndarray, empty_value: float = 1.0) -> float:
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as im1. If not boolean, it will be converted.
    empty_value : scalar, float.

    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        If both images are empty (sum equal to zero) = empty_value
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return float(empty_value)

    intersection = np.logical_and(im1, im2)

    return float(2.0 * intersection.sum() / im_sum)


def compute_absolute_volume_difference(im1: np.ndarray, im2: np.ndarray, voxel_size: float) -> float:
    """
    Computes the absolute volume difference between two masks.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_size : scalar, float (ml)
        If not float, it will be converted.

    Returns
    -------
    abs_vol_diff : float, measured in ml.
        Absolute volume difference as a float.
        Maximum similarity = 0
        No similarity = inf
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    voxel_size = float(voxel_size)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have different shapes."
            " The absolute volume difference is computed with mismatching shape masks"
        )

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size
    abs_vol_diff = np.abs(ground_truth_volume - prediction_volume)

    return float(abs_vol_diff)


def compute_absolute_lesion_difference(ground_truth: np.ndarray, prediction: np.ndarray, connectivity: int = 26) -> int:
    """
    Computes the absolute lesion difference between two masks. The number of lesions are counted for
    each volume, and their absolute difference is computed.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.

    Returns
    -------
    abs_les_diff : int
        Absolute lesion difference as integer.
        Maximum similarity = 0
        No similarity = inf
    """
    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    _, ground_truth_numb_lesion = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
    _, prediction_numb_lesion = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
    abs_les_diff = abs(ground_truth_numb_lesion - prediction_numb_lesion)

    return int(abs_les_diff)


def compute_lesion_f1_score(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    empty_value: float = 1.0,
    connectivity: int = 26,
) -> float:
    """
    Computes the lesion-wise F1-score between two masks.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    empty_value : scalar, float.
    connectivity : scalar, int.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value
    """
    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    tp = 0
    fp = 0
    fn = 0

    intersection = np.logical_and(ground_truth, prediction)
    labeled_ground_truth, N = cc3d.connected_components(
        ground_truth, connectivity=connectivity, return_N=True
    )

    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
            if np.logical_and(binary_cluster_image, intersection).any():
                tp += 1
            else:
                fn += 1

    labeled_prediction, N = cc3d.connected_components(
        prediction, connectivity=connectivity, return_N=True
    )
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_prediction, binary=True, in_place=True):
            if not np.logical_and(binary_cluster_image, ground_truth).any():
                fp += 1

    f1_score = empty_value
    if tp + fp + fn != 0:
        f1_score = tp / (tp + (fp + fn) / 2)

    return float(f1_score)


def load_nifti_mask(mask_path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load NIfTI mask and extract voxel spacing.

    Args:
        mask_path: Path to NIfTI file

    Returns:
        Tuple of (mask array, voxel spacing)
    """
    nii = nib.load(str(mask_path))
    mask = nii.get_fdata()

    # Get voxel spacing from header
    voxel_spacing = tuple(nii.header.get_zooms()[:3])

    return mask, voxel_spacing


# ===========================
# Main Grading Function
# ===========================

def compute_metrics_single_case(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
) -> Dict[str, float]:
    """
    Compute all metrics for a single case.

    Args:
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        voxel_spacing: Voxel spacing in mm

    Returns:
        Dictionary of metrics
    """
    dice = compute_dice(pred_mask, gt_mask)
    lesion_f1 = compute_lesion_f1_score(gt_mask, pred_mask)
    lesion_count_diff = compute_absolute_lesion_difference(gt_mask, pred_mask)
    voxel_volume_ml = float(np.prod(voxel_spacing) / 1000.0)  # convert mm^3 to ml
    abs_volume_diff = compute_absolute_volume_difference(gt_mask, pred_mask, voxel_volume_ml)

    return {
        'dice': dice,
        'lesion_f1': lesion_f1,
        'lesion_count_diff': lesion_count_diff,
        'abs_volume_diff': abs_volume_diff,
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame,
          submission_dir: Optional[Path] = None,
          answers_dir: Optional[Path] = None) -> dict:
    """
    Grade ISLES'22 submission using lesion segmentation metrics (Dice, lesion F1,
    lesion count difference, absolute volume difference).

    Args:
        submission: Submission DataFrame with columns: case_id, predicted_mask_path
        answers: Ground truth DataFrame with columns: case_id, mask_path
        submission_dir: Directory containing submission files
        answers_dir: Directory containing ground truth files

    Returns:
        Dictionary with all metrics
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

    # Check IDs match
    submission_ids = set(submission['case_id'].astype(str))
    answer_ids = set(answers['case_id'].astype(str))

    if submission_ids != answer_ids:
        missing = answer_ids - submission_ids
        extra = submission_ids - answer_ids
        error_msg = "Submission IDs don't match ground truth"
        if missing:
            error_msg += f"\n  Missing IDs: {sorted(list(missing))[:5]}"
        if extra:
            error_msg += f"\n  Extra IDs: {sorted(list(extra))[:5]}"
        raise InvalidSubmissionError(error_msg)

    # Compute metrics for each case
    all_metrics = []
    failed_cases = []

    for _, row in submission.iterrows():
        case_id = str(row['case_id'])
        pred_mask_path = submission_dir / row['predicted_mask_path']

        # Find ground truth
        gt_row = answers[answers['case_id'].astype(str) == case_id]
        if gt_row.empty:
            raise InvalidSubmissionError(f"No ground truth found for {case_id}")

        gt_mask_path = answers_dir / gt_row.iloc[0]['mask_path']

        # Validate files exist
        if not pred_mask_path.exists():
            raise InvalidSubmissionError(f"Predicted mask not found: {pred_mask_path}")
        if not gt_mask_path.exists():
            raise InvalidSubmissionError(f"Ground truth mask not found: {gt_mask_path}")

        try:
            # Load masks
            gt_mask, voxel_spacing = load_nifti_mask(gt_mask_path)
            pred_mask, _ = load_nifti_mask(pred_mask_path)

            # Resize prediction if shapes don't match (use nearest neighbor for segmentation masks)
            if gt_mask.shape != pred_mask.shape:
                from scipy.ndimage import zoom
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Shape mismatch for {case_id}: "
                    f"predicted {pred_mask.shape} vs ground truth {gt_mask.shape}. "
                    f"Resizing prediction to match ground truth."
                )
                # Calculate zoom factors for each dimension
                zoom_factors = [gt_dim / pred_dim for gt_dim, pred_dim in zip(gt_mask.shape, pred_mask.shape)]
                # Use order=0 for nearest neighbor interpolation (appropriate for segmentation masks)
                pred_mask = zoom(pred_mask, zoom_factors, order=0).astype(pred_mask.dtype)

            # Compute metrics
            metrics = compute_metrics_single_case(gt_mask, pred_mask, voxel_spacing)
            metrics['case_id'] = case_id
            all_metrics.append(metrics)

        except Exception as e:
            failed_cases.append(case_id)
            raise InvalidSubmissionError(f"Error evaluating {case_id}: {str(e)}")

    # Aggregate results
    if len(all_metrics) == 0:
        raise InvalidSubmissionError("No metrics computed")

    metrics_df = pd.DataFrame(all_metrics)

    mean_dice = float(metrics_df['dice'].mean())
    mean_lesion_f1 = float(metrics_df['lesion_f1'].mean())
    mean_lesion_count_diff = float(metrics_df['lesion_count_diff'].mean())
    mean_abs_volume_diff = float(metrics_df['abs_volume_diff'].mean())

    # Print summary
    print(f"\n=== ISLES'22 Stroke Lesion Segmentation Results ===")
    print(f"Cases evaluated: {len(all_metrics)}")
    print(f"Failed cases: {len(failed_cases)}")
    print(f"\nMetrics:")
    print(f"  Mean Dice Score: {mean_dice:.4f}")
    print(f"  Mean Lesion F1: {mean_lesion_f1:.4f}")
    print(f"  Mean Lesion Count Diff: {mean_lesion_count_diff:.2f}")
    print(f"  Mean Absolute Volume Diff (ml): {mean_abs_volume_diff:.4f}")

    results_dict = {
        "dice": mean_dice,
        "lesion_f1": mean_lesion_f1,
        "lesion_count_diff": mean_lesion_count_diff,
        "abs_volume_diff": mean_abs_volume_diff,
    }

    # Calculate mean_position if leaderboard exists
    # Note: Leaderboard may have different metric names, so this is a best-effort
    leaderboard_path = Path(__file__).parent / 'leaderboard.csv'
    if leaderboard_path.exists():
        leaderboard_df = pd.read_csv(leaderboard_path)
        metric_cols = [col for col in leaderboard_df.columns
                        if not col.endswith('_position') and col not in ['rank', 'team', 'mean_position']]

        submission_dict = {k: v for k, v in results_dict.items() if k in metric_cols}
        if submission_dict:
            positions = calculate_positions_and_mean(submission_dict, leaderboard_df)
            mean_position_value = positions.get('mean_position')
            print(f"Positions: {positions}")
            print(f"Mean position: {mean_position_value:.2f}")
            if mean_position_value is not None:
                # Add individual metric positions to results
                for metric_name, position_value in positions.items():
                    if metric_name != 'mean_position':
                        results_dict[metric_name] = position_value

                results_dict['mean_position'] = mean_position_value

                num_competitors = len(leaderboard_df)
                percentile = 1 - ((mean_position_value - 1) / (num_competitors))
                results_dict['percentile'] = percentile

                results_dict['overall'] = mean_position_value

    if 'overall' not in results_dict:
        results_dict['overall'] = mean_dice  # Fallback to primary metric

    print(f"Results dict: {results_dict}")
    return results_dict



# Metric direction configuration and position calculation
METRIC_DIRECTIONS = {
    "dice": "higher_better",
    "lesion_f1": "higher_better",
    "lesion_count_diff": "lower_better",
    "abs_volume_diff": "lower_better"
}

def calculate_positions_and_mean(submission_metrics, leaderboard_df):
    """
    Calculate position for each metric and overall mean_position.
    
    Args:
        submission_metrics: Dict of metric values for the submission
        leaderboard_df: DataFrame containing all leaderboard entries
    
    Returns:
        Dict with metric positions and mean_position
    """
    import pandas as pd
    
    # Add submission to leaderboard temporarily for ranking
    temp_df = pd.concat([leaderboard_df, pd.DataFrame([submission_metrics])], ignore_index=True)

    positions = {}
    metric_columns = [col for col in submission_metrics.keys()
                     if col in METRIC_DIRECTIONS and not col.endswith('_position')]

    for metric in metric_columns:
        if metric not in temp_df.columns:
            continue

        direction = METRIC_DIRECTIONS.get(metric, 'higher_better')

        if direction == 'lower_better':
            # Lower values get better (lower) positions
            temp_df[f'{metric}_position'] = temp_df[metric].rank(ascending=True, method='min')
        else:
            # Higher values get better (lower) positions
            temp_df[f'{metric}_position'] = temp_df[metric].rank(ascending=False, method='min')

        # Get position for the submitted entry (last row)
        positions[f'{metric}_position'] = temp_df[f'{metric}_position'].iloc[-1]
    
    # Calculate mean position
    if positions:
        positions['mean_position'] = sum(positions.values()) / len(positions)
    else:
        positions['mean_position'] = None
    
    return positions
