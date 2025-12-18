"""
Grading script for NeurIPS 2022 Cell Segmentation Challenge

Implements F1 score metric based on instance segmentation matching.
Adapted from: https://github.com/JunMa11/NeurIPS-CellSeg/blob/main/baseline/compute_metric.py

Evaluation Metrics:
1. F1 Score (primary metric, IoU threshold = 0.5)
2. Precision: TP / (TP + FP)
3. Recall: TP / (TP + FN)
4. Dice coefficient: Binary segmentation quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from numba import jit
from scipy.optimize import linear_sum_assignment
from skimage import segmentation
import tifffile as tif
from PIL import Image

from rexmle.grade_helpers import InvalidSubmissionError


# Metric directions for position calculation
METRIC_DIRECTIONS = {
    'f1_0.5': 'higher_better',
    'f1_0.6': 'higher_better',
    'f1_0.7': 'higher_better',
    'f1_0.8': 'higher_better',
    'f1_0.9': 'higher_better',
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


# ===========================
# Constants
# ===========================

# IoU thresholds for evaluation
IOU_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

# Image formats supported
IMAGE_FORMATS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']


# ===========================
# Core Metric Functions
# ===========================

@jit(nopython=True)
def _label_overlap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fast function to get pixel overlaps between masks in x and y.

    Parameters:
        x: ND-array, int - where 0=NO masks; 1,2... are mask labels
        y: ND-array, int - where 0=NO masks; 1,2... are mask labels

    Returns:
        overlap: ND-array, int - matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    """
    x = x.ravel()
    y = y.ravel()

    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    for i in range(len(x)):
        overlap[x[i], y[i]] += 1

    return overlap


def _intersection_over_union(masks_true: np.ndarray, masks_pred: np.ndarray) -> np.ndarray:
    """
    Intersection over union of all mask pairs.

    Parameters:
        masks_true: ND-array, int - ground truth masks, where 0=NO masks; 1,2... are mask labels
        masks_pred: ND-array, int - predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns:
        iou: ND-array, float - IoU matrix for all pairs
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou: np.ndarray, th: float) -> int:
    """
    True positive at threshold th using Hungarian algorithm.

    Parameters:
        iou: float, ND-array - array of IOU pairs
        th: float - threshold on IOU for positive label

    Returns:
        tp: int - number of true positives at threshold
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return int(tp)


def eval_tp_fp_fn(masks_true: np.ndarray, masks_pred: np.ndarray,
                  threshold: float = 0.5) -> Tuple[int, int, int]:
    """
    Evaluate true positives, false positives, and false negatives.

    Parameters:
        masks_true: Ground truth instance masks
        masks_pred: Predicted instance masks
        threshold: IoU threshold for matching

    Returns:
        tp: True positives
        fp: False positives
        fn: False negatives
    """
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)

    if num_inst_seg > 0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
        tp = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        tp = 0
        fp = 0
        fn = num_inst_gt

    return tp, fp, fn


def dice(gt: np.ndarray, seg: np.ndarray) -> float:
    """
    Compute Dice coefficient for binary masks.

    Parameters:
        gt: Ground truth binary mask
        seg: Predicted binary mask

    Returns:
        dice_score: Dice coefficient
    """
    if np.count_nonzero(gt) == 0 and np.count_nonzero(seg) == 0:
        dice_score = 1.0
    elif np.count_nonzero(gt) == 0 and np.count_nonzero(seg) > 0:
        dice_score = 0.0
    else:
        union = np.count_nonzero(np.logical_and(gt, seg))
        intersection = np.count_nonzero(gt) + np.count_nonzero(seg)
        dice_score = 2 * union / intersection
    return dice_score


def remove_boundary_cells(mask: np.ndarray, margin: int = 2) -> np.ndarray:
    """
    Remove boundary cells from evaluation.
    Cells touching image borders within margin pixels are set to background.

    Parameters:
        mask: Instance segmentation mask
        margin: Pixel margin from border

    Returns:
        new_label: Mask with boundary cells removed and relabeled
    """
    H, W = mask.shape
    bd = np.ones((H, W))
    bd[margin:H-margin, margin:W-margin] = 0
    bd_cells = np.unique(mask * bd)

    # Remove boundary cells
    for i in bd_cells[1:]:  # Skip background (0)
        mask[mask == i] = 0

    # Relabel sequentially
    new_label, _, _ = segmentation.relabel_sequential(mask)
    return new_label


def load_mask(mask_path: Path) -> np.ndarray:
    """
    Load segmentation mask from file.

    Parameters:
        mask_path: Path to mask file

    Returns:
        mask: Loaded mask as numpy array
    """
    suffix = mask_path.suffix.lower()

    if suffix in ['.tif', '.tiff']:
        mask = tif.imread(str(mask_path))
    else:
        mask = np.array(Image.open(mask_path))

    # Ensure integer type
    return mask.astype(np.int32)


# ===========================
# Main Grading Function
# ===========================

def compute_metrics_single_case(gt_mask: np.ndarray, pred_mask: np.ndarray,
                                 threshold: float = 0.5,
                                 remove_boundary: bool = True) -> Dict[str, float]:
    """
    Compute all metrics for a single case.

    Parameters:
        gt_mask: Ground truth instance mask
        pred_mask: Predicted instance mask
        threshold: IoU threshold for matching
        remove_boundary: Whether to remove boundary cells

    Returns:
        metrics: Dictionary of computed metrics
    """
    # Compute binary Dice
    dice_score = dice(gt_mask > 0, pred_mask > 0)

    # Handle large images (>25M pixels) with patch-based evaluation
    if np.prod(gt_mask.shape) < 25000000:
        # Small/medium images - evaluate directly
        if remove_boundary:
            gt_mask = remove_boundary_cells(gt_mask.astype(np.int32))
            pred_mask = remove_boundary_cells(pred_mask.astype(np.int32))

        gt_mask, _, _ = segmentation.relabel_sequential(gt_mask)
        pred_mask, _, _ = segmentation.relabel_sequential(pred_mask)

        cell_true_num = np.max(gt_mask)
        cell_pred_num = np.max(pred_mask)

        tp, fp, fn = eval_tp_fp_fn(gt_mask, pred_mask, threshold=threshold)

    else:
        # Large images - use patch-based evaluation
        H, W = gt_mask.shape
        roi_size = 2000

        # Calculate padding
        if H % roi_size != 0:
            n_H = H // roi_size + 1
            new_H = roi_size * n_H
        else:
            n_H = H // roi_size
            new_H = H

        if W % roi_size != 0:
            n_W = W // roi_size + 1
            new_W = roi_size * n_W
        else:
            n_W = W // roi_size
            new_W = W

        # Pad images
        gt_pad = np.zeros((new_H, new_W), dtype=gt_mask.dtype)
        pred_pad = np.zeros((new_H, new_W), dtype=gt_mask.dtype)
        gt_pad[:H, :W] = gt_mask
        pred_pad[:H, :W] = pred_mask

        tp = 0
        fp = 0
        fn = 0
        cell_true_num = 0
        cell_pred_num = 0

        # Evaluate each ROI
        for i in range(n_H):
            for j in range(n_W):
                gt_roi = gt_pad[roi_size*i:roi_size*(i+1), roi_size*j:roi_size*(j+1)]
                pred_roi = pred_pad[roi_size*i:roi_size*(i+1), roi_size*j:roi_size*(j+1)]

                if remove_boundary:
                    gt_roi = remove_boundary_cells(gt_roi.copy())
                    pred_roi = remove_boundary_cells(pred_roi.copy())

                gt_roi, _, _ = segmentation.relabel_sequential(gt_roi)
                pred_roi, _, _ = segmentation.relabel_sequential(pred_roi)

                cell_true_num += np.max(gt_roi)
                cell_pred_num += np.max(pred_roi)

                tp_i, fp_i, fn_i = eval_tp_fp_fn(gt_roi, pred_roi, threshold=threshold)
                tp += tp_i
                fp += fp_i
                fn += fn_i

    # Compute precision, recall, and F1
    if tp == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = tp / cell_pred_num if cell_pred_num > 0 else 0.0
        recall = tp / cell_true_num if cell_true_num > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'true_num': cell_true_num,
        'pred_num': cell_pred_num,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'dice': dice_score,
        'f1': f1
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame,
          submission_dir: Optional[Path] = None,
          answers_dir: Optional[Path] = None) -> dict:
    """
    Grade NeurIPS Cell Segmentation submission using F1 score.

    Args:
        submission: Submission DataFrame with columns: image_id, predicted_mask_path
        answers: Ground truth DataFrame with columns: image_id, label_path
        submission_dir: Directory containing submission files
        answers_dir: Directory containing ground truth files

    Returns:
        Dictionary with F1 score and other metrics
    """
    if submission_dir is None:
        raise InvalidSubmissionError("Segmentation task requires submission_dir")

    if answers_dir is None:
        raise InvalidSubmissionError("Segmentation task requires answers_dir")

    # Validate submission format
    if 'image_id' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'image_id' column")
    if 'predicted_mask_path' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'predicted_mask_path' column")

    # Check IDs match
    submission_ids = set(submission['image_id'].astype(str))
    answer_ids = set(answers['image_id'].astype(str))

    if submission_ids != answer_ids:
        missing = answer_ids - submission_ids
        extra = submission_ids - answer_ids
        error_msg = "Submission IDs don't match ground truth"
        if missing:
            error_msg += f"\n  Missing IDs: {sorted(list(missing))[:5]}"
        if extra:
            error_msg += f"\n  Extra IDs: {sorted(list(extra))[:5]}"
        raise InvalidSubmissionError(error_msg)

    # Compute metrics for each case at all thresholds
    all_metrics_by_threshold = {th: [] for th in IOU_THRESHOLDS}
    failed_cases = []

    for _, row in submission.iterrows():
        image_id = str(row['image_id'])
        pred_mask_path = submission_dir / row['predicted_mask_path']

        # Find ground truth
        gt_row = answers[answers['image_id'].astype(str) == image_id]
        if gt_row.empty:
            raise InvalidSubmissionError(f"No ground truth found for {image_id}")

        gt_mask_path = answers_dir / gt_row.iloc[0]['label_path']

        # Validate files exist
        if not pred_mask_path.exists():
            raise InvalidSubmissionError(f"Predicted mask not found: {pred_mask_path}")
        if not gt_mask_path.exists():
            raise InvalidSubmissionError(f"Ground truth mask not found: {gt_mask_path}")

        try:
            # Load masks
            gt_mask = load_mask(gt_mask_path)
            pred_mask = load_mask(pred_mask_path)

            # Resize prediction if shapes don't match (use nearest neighbor for instance masks)
            if gt_mask.shape != pred_mask.shape:
                from PIL import Image as PILImage
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Shape mismatch for {image_id}: "
                    f"predicted {pred_mask.shape} vs ground truth {gt_mask.shape}. "
                    f"Resizing prediction to match ground truth."
                )
                # Resize prediction to match ground truth shape
                pred_pil = PILImage.fromarray(pred_mask.astype(np.uint16))
                pred_pil_resized = pred_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), PILImage.NEAREST)
                pred_mask = np.array(pred_pil_resized, dtype=np.int32)

            # Compute metrics at all thresholds
            for threshold in IOU_THRESHOLDS:
                metrics = compute_metrics_single_case(
                    gt_mask, pred_mask,
                    threshold=threshold,
                    remove_boundary=True
                )
                metrics['image_id'] = image_id
                all_metrics_by_threshold[threshold].append(metrics)

        except Exception as e:
            failed_cases.append(image_id)
            raise InvalidSubmissionError(f"Error evaluating {image_id}: {str(e)}")

    # Aggregate results
    if len(all_metrics_by_threshold[0.5]) == 0:
        raise InvalidSubmissionError("No metrics computed")

    # Compute mean F1 for each threshold
    f1_scores = {}
    for threshold in IOU_THRESHOLDS:
        metrics_df = pd.DataFrame(all_metrics_by_threshold[threshold])
        mean_f1 = float(metrics_df['f1'].mean())
        f1_scores[f'f1_{threshold}'] = mean_f1

    # Get additional metrics from primary threshold (0.5)
    primary_metrics_df = pd.DataFrame(all_metrics_by_threshold[0.5])
    mean_precision = float(primary_metrics_df['precision'].mean())
    mean_recall = float(primary_metrics_df['recall'].mean())
    mean_dice = float(primary_metrics_df['dice'].mean())
    median_f1 = float(primary_metrics_df['f1'].median())

    total_tp = int(primary_metrics_df['tp'].sum())
    total_fp = int(primary_metrics_df['fp'].sum())
    total_fn = int(primary_metrics_df['fn'].sum())

    # Print summary
    print(f"\n=== NeurIPS Cell Segmentation Results ===")
    print(f"Cases evaluated: {len(all_metrics_by_threshold[0.5])}")
    print(f"Failed cases: {len(failed_cases)}")
    print(f"\nF1 Scores at Different IoU Thresholds:")
    for threshold in IOU_THRESHOLDS:
        print(f"  F1 @ {threshold}: {f1_scores[f'f1_{threshold}']:.4f}")
    print(f"\nAdditional Metrics (IoU = 0.5):")
    print(f"  Mean Precision: {mean_precision:.4f}")
    print(f"  Mean Recall: {mean_recall:.4f}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"  Median F1: {median_f1:.4f}")

    # Build results dictionary with only F1 scores at different thresholds
    results_dict = f1_scores.copy()

    # Calculate position and mean position if leaderboard exists
    mean_position_value = None
    leaderboard_path = Path(__file__).parent / 'leaderboard.csv'

    if leaderboard_path.exists():
        try:
            leaderboard_df = pd.read_csv(leaderboard_path)
            # Use only the F1 scores at different thresholds for position comparison
            submission_dict = {
                "f1_0.5": results_dict["f1_0.5"],
                "f1_0.6": results_dict["f1_0.6"],
                "f1_0.7": results_dict["f1_0.7"],
                "f1_0.8": results_dict["f1_0.8"],
                "f1_0.9": results_dict["f1_0.9"],
            }
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

            print(f"\nMean Position: {mean_position_value:.2f}")
            print(f"Percentile: {percentile:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate position: {e}")

    # Set 'overall' for ranking (lower mean_position is better)
    if mean_position_value is not None:
        results_dict['overall'] = mean_position_value
    else:
        # Fallback: use negative F1@0.5 for ranking (higher F1 = lower/better rank)
        results_dict['overall'] = -results_dict['f1_0.5']

    return results_dict
