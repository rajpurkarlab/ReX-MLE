"""
Grading script for SEG.A. 2023 - Segmentation of the Aortic Vessel Tree

Evaluation Metrics:
1. Dice Similarity Coefficient (DSC) - Primary metric for ranking
2. Hausdorff Distance (HD) - Secondary metric for ranking

Note: The full SEG.A. challenge uses Sobol' sensitivity analysis with image variations
(rotations, translations, intensity, noise, blur) to compute a comprehensive ranking.
This implementation computes the base DSC and HD metrics on the original test set.

Full ranking formula (from challenge):

p1 = 1 - Σ|Si^(1) - 1/M| where M is number of image variations
   (Measures uniformity of first-order Sobol' indices)

p2 = Σ(Si^(T) - Si^(1)) where T=total-order, (1)=first-order
   (Measures interaction between variations)

p3 = 0.6 * r_medianHD + 0.25 * r_varHD + 0.15 * r_skewHD
   (HD distribution: lower median, lower variance, positive skewness is better)

p4 = 0.6 * r_medianDSC + 0.25 * r_varDSC + 0.15 * r_skewDSC
   (DSC distribution: median closer to 1, lower variance, negative skewness is better)

r_fin = (r1 + r2)/6 + (r3 + r4)/3
   where r1, r2, r3, r4 are ranks based on p1, p2, p3, p4
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from joblib import Parallel, delayed
from tqdm import tqdm
from monai.metrics.utils import get_mask_edges, get_surface_distance
from scipy.stats import skew
import os

from rexmle.grade_helpers import InvalidSubmissionError
from rexmle.medical_utils import load_medical_image


# Metric directions for position calculation
METRIC_DIRECTIONS = {
    'hd_50pc': 'lower_better',
    'dsc_50pc': 'higher_better',
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


def compute_dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Dice Similarity Coefficient for binary segmentation.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask

    Returns:
        Dice coefficient (0 to 1, higher is better)
    """
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()

    if gt_sum == 0 and pred_sum == 0:
        return 1.0  # Both empty, perfect match

    if gt_sum == 0:
        return 0.0  # False positives only

    # Standard Dice formula: 2*|A ∩ B| / (|A| + |B|)
    dice = 2.0 * intersection / (pred_sum + gt_sum)

    return float(dice)


def compute_hausdorff_distance(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: Tuple = None) -> float:
    """
    Compute Hausdorff Distance between predicted and ground truth surfaces.

    The Hausdorff distance measures the maximum distance from any point on one
    surface to the closest point on the other surface.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        spacing: Voxel spacing (z, y, x) for physical distance calculation

    Returns:
        Hausdorff distance (lower is better, 0 is perfect)
    """
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    # Handle cases where both masks are empty
    if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0:
        return 0.0

    # Handle cases where one mask is empty and the other is not
    if np.sum(pred_binary) == 0 or np.sum(gt_binary) == 0:
        shape = pred_mask.shape
        if spacing is not None:
            diagonal = np.sqrt(sum((s * d) ** 2 for s, d in zip(shape, spacing)))
        else:
            diagonal = np.sqrt(sum(s ** 2 for s in shape))
        return float(diagonal)

    # If masks are identical and not empty, HD is 0
    if np.all(pred_binary == gt_binary):
        return 0.0

    # Compute surface distances using MONAI utilities
    edges_pred, edges_gt = get_mask_edges(pred_binary, gt_binary)

    # get_surface_distance computes distances from edges_pred to edges_gt and vice versa
    # and returns a concatenated array of these distances.
    # The Hausdorff distance is the maximum of these surface distances.
    surface_distances = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean", spacing=spacing)

    # If surface_distances is empty, it means there are no edges to compare,
    # but we've already handled cases where masks are identical or one/both are empty.
    # This case should ideally not be reached if the above checks are comprehensive.
    # However, as a safeguard, if it's empty, it implies no distance, so 0.0.
    if surface_distances.shape == (0,):
        return 0.0

    hausdorff_dist = surface_distances.max()

    return float(hausdorff_dist)


def compute_distribution_metrics(values: np.ndarray) -> Dict[str, float]:
    """
    Compute distribution statistics for DSC or HD values.

    Args:
        values: Array of metric values across all test cases

    Returns:
        Dictionary with median, variance, and skewness
    """
    return {
        'median': float(np.median(values)),
        'variance': float(np.var(values)),
        'skewness': float(skew(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values))
    }


def compute_p3_score(hd_stats: Dict[str, float], all_hd_stats: list = None) -> float:
    """
    Compute p3 score for Hausdorff Distance distribution.

    p3 = 0.6 * r_medianHD + 0.25 * r_varHD + 0.15 * r_skewHD

    For HD:
    - Lower median is better (closer to 0)
    - Lower variance is better (more consistent)
    - Positive skewness is better (most values are low)

    Args:
        hd_stats: HD statistics for current submission
        all_hd_stats: List of HD statistics from all submissions for ranking
                      If None, returns the raw component values

    Returns:
        p3 score (if all_hd_stats is None) or weighted rank
    """
    if all_hd_stats is None:
        # Return raw values for single submission
        return {
            'median': hd_stats['median'],
            'variance': hd_stats['variance'],
            'skewness': hd_stats['skewness']
        }

    # Compute ranks (lower rank = better performance)
    medians = [s['median'] for s in all_hd_stats]
    variances = [s['variance'] for s in all_hd_stats]
    skewnesses = [s['skewness'] for s in all_hd_stats]

    # Rank median: lower is better (ascending)
    r_median = np.argsort(np.argsort(medians)) + 1
    # Rank variance: lower is better (ascending)
    r_var = np.argsort(np.argsort(variances)) + 1
    # Rank skewness: higher is better (descending), so negate for sorting
    r_skew = np.argsort(np.argsort(-np.array(skewnesses))) + 1

    # Get current submission's rank
    current_idx = len(all_hd_stats) - 1  # Assumes current is last

    # Weighted combination
    p3 = 0.6 * r_median[current_idx] + 0.25 * \
        r_var[current_idx] + 0.15 * r_skew[current_idx]

    return float(p3)


def compute_p4_score(dsc_stats: Dict[str, float], all_dsc_stats: list = None) -> float:
    """
    Compute p4 score for DSC distribution.

    p4 = 0.6 * r_medianDSC + 0.25 * r_varDSC + 0.15 * r_skewDSC

    For DSC:
    - Higher median is better (closer to 1)
    - Lower variance is better (more consistent)
    - Negative skewness is better (most values are high)

    Args:
        dsc_stats: DSC statistics for current submission
        all_dsc_stats: List of DSC statistics from all submissions for ranking
                       If None, returns the raw component values

    Returns:
        p4 score (if all_dsc_stats is None) or weighted rank
    """
    if all_dsc_stats is None:
        # Return raw values for single submission
        return {
            'median': dsc_stats['median'],
            'variance': dsc_stats['variance'],
            'skewness': dsc_stats['skewness']
        }

    # Compute ranks (lower rank = better performance)
    medians = [s['median'] for s in all_dsc_stats]
    variances = [s['variance'] for s in all_dsc_stats]
    skewnesses = [s['skewness'] for s in all_dsc_stats]

    # Rank median: higher is better (descending), so negate
    r_median = np.argsort(np.argsort(-np.array(medians))) + 1
    # Rank variance: lower is better (ascending)
    r_var = np.argsort(np.argsort(variances)) + 1
    # Rank skewness: more negative is better, so use as-is for sorting (lower is better)
    r_skew = np.argsort(np.argsort(skewnesses)) + 1

    # Get current submission's rank
    current_idx = len(all_dsc_stats) - 1  # Assumes current is last

    # Weighted combination
    p4 = 0.6 * r_median[current_idx] + 0.25 * \
        r_var[current_idx] + 0.15 * r_skew[current_idx]

    return float(p4)


def _compute_metrics_for_case(
    row: pd.Series,
    submission_dir: Path,
    answers: pd.DataFrame,
    answers_dir: Path
) -> Dict[str, Any]:
    """Helper function to compute metrics for a single case."""
    image_id = row['image_id']
    pred_mask_path = submission_dir / row['predicted_mask_path']

    # Find ground truth
    gt_row = answers[answers['image_id'] == image_id]
    if gt_row.empty:
        return {'error': f"No ground truth found for {image_id}"}

    gt_mask_path = answers_dir / gt_row.iloc[0]['label_path']

    if not pred_mask_path.exists():
        return {'error': f"Predicted mask not found: {pred_mask_path}"}

    try:
        pred_data = load_medical_image(pred_mask_path)
        pred_mask = pred_data['image']
        pred_spacing = pred_data.get('spacing', None)
    except Exception as e:
        return {'error': f"Error loading predicted mask {pred_mask_path}: {e}"}

    try:
        gt_data = load_medical_image(gt_mask_path)
        gt_mask = gt_data['image']
        gt_spacing = gt_data.get('spacing', None)
    except Exception as e:
        return {'error': f"Error loading ground truth mask {gt_mask_path}: {e}"}

    # Resize prediction if shapes don't match (use nearest neighbor for segmentation masks)
    if pred_mask.shape != gt_mask.shape:
        from scipy.ndimage import zoom
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Shape mismatch for {image_id}: "
            f"predicted {pred_mask.shape} vs ground truth {gt_mask.shape}. "
            f"Resizing prediction to match ground truth."
        )
        # Calculate zoom factors for each dimension
        zoom_factors = [gt_dim / pred_dim for gt_dim, pred_dim in zip(gt_mask.shape, pred_mask.shape)]
        # Use order=0 for nearest neighbor interpolation (appropriate for segmentation masks)
        pred_mask = zoom(pred_mask, zoom_factors, order=0).astype(pred_mask.dtype)

    spacing = gt_spacing if gt_spacing is not None else pred_spacing

    dice = compute_dice_coefficient(pred_mask, gt_mask)
    hausdorff = compute_hausdorff_distance(pred_mask, gt_mask, spacing)

    return {
        'image_id': image_id,
        'dice': dice,
        'hausdorff_distance': hausdorff
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame, submission_dir: Optional[Path] = None, answers_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Grade a SEG.A. submission.

    Args:
        submission: Submission DataFrame with columns: image_id, predicted_mask_path
        answers: Ground truth DataFrame with columns: image_id, image_path, label_path
        submission_dir: Directory containing submission files
        answers_dir: Directory containing ground truth files

    Returns:
        Dictionary with metrics matching leaderboard columns:
        - mean_position: Mean position across metrics
        - hd_50pc: Median (50th percentile) Hausdorff Distance
        - dsc_50pc: Median (50th percentile) Dice Similarity Coefficient
        - overall: Mean position (primary metric for ranking)

    Notes:
        The full SEG.A. challenge ranking includes:
        - Sobol' sensitivity analysis on image variations
        - Distribution analysis of DSC and HD
        - Weighted ranking combining multiple factors

        This implementation returns mean DSC as the primary metric, with HD as auxiliary.
    """
    if submission_dir is None:
        raise InvalidSubmissionError(
            "Segmentation task requires submission_dir")

    if answers_dir is None:
        raise InvalidSubmissionError(
            "Segmentation task requires answers_dir")

    # Validate submission format
    if 'image_id' not in submission.columns:
        raise InvalidSubmissionError(
            "Submission must contain 'image_id' column")

    if 'predicted_mask_path' not in submission.columns:
        raise InvalidSubmissionError(
            "Submission must contain 'predicted_mask_path' column")

    # Check answers format
    assert 'image_id' in answers.columns, "Answers must contain 'image_id' column"
    assert 'label_path' in answers.columns, "Answers must contain 'label_path' column"

    # Check IDs match
    submission_ids = set(submission['image_id'])
    answer_ids = set(answers['image_id'])

    if submission_ids != answer_ids:
        missing = answer_ids - submission_ids
        extra = submission_ids - answer_ids

        error_msg = "Submission IDs don't match ground truth"
        if missing:
            error_msg += f"\n  Missing IDs: {list(missing)[:5]}..."
        if extra:
            error_msg += f"\n  Extra IDs: {list(extra)[:5]}..."

        raise InvalidSubmissionError(error_msg)

    # Compute metrics for each case in parallel
    print(f"\nEvaluating {len(submission)} segmentation masks in parallel...")
    
    tasks = [
        delayed(_compute_metrics_for_case)(row, submission_dir, answers, answers_dir)
        for _, row in submission.iterrows()
    ]

    # Use n_jobs=-1 to use all available CPU cores
    results = Parallel(n_jobs=-1)(tqdm(tasks, desc="Processing cases"))

    # Process results and check for errors
    all_metrics = []
    for res in results:
        if isinstance(res, dict) and 'error' in res:
            raise InvalidSubmissionError(res['error'])
        all_metrics.append(res)

    all_dice_scores = [m['dice'] for m in all_metrics]
    all_hausdorff_distances = [m['hausdorff_distance'] for m in all_metrics]

    # Convert to arrays for statistical analysis
    dice_array = np.array(all_dice_scores)
    hd_array = np.array(all_hausdorff_distances)

    # Compute distribution metrics
    dice_stats = compute_distribution_metrics(dice_array)
    hd_stats = compute_distribution_metrics(hd_array)

    # Compute p3 and p4 component values (for single submission, returns dict)
    p3_components = compute_p3_score(hd_stats, all_hd_stats=None)
    p4_components = compute_p4_score(dice_stats, all_dsc_stats=None)

    # Primary metric for ranking: mean DSC
    mean_dice = dice_stats['mean']

    # Print detailed results
    print(f"\n{'=' * 70}")
    print(f"SEG.A. 2023 - Aortic Vessel Tree Segmentation Results")
    print(f"{'=' * 70}")
    print(f"\nCases evaluated: {len(all_metrics)}")

    print(f"\nDice Similarity Coefficient (DSC):")
    print(f"  Mean:     {dice_stats['mean']:.4f}")
    print(f"  Median:   {dice_stats['median']:.4f}")
    print(f"  Std Dev:  {dice_stats['std']:.4f}")
    print(f"  Variance: {dice_stats['variance']:.4f}")
    print(f"  Skewness: {dice_stats['skewness']:.4f}")

    print(f"\nHausdorff Distance (HD) [mm]:")
    print(f"  Mean:     {hd_stats['mean']:.2f}")
    print(f"  Median:   {hd_stats['median']:.2f}")
    print(f"  Std Dev:  {hd_stats['std']:.2f}")
    print(f"  Variance: {hd_stats['variance']:.2f}")
    print(f"  Skewness: {hd_stats['skewness']:.4f}")

    print(f"\n{'=' * 70}")
    print(f"Challenge Ranking Components (for multi-submission comparison):")
    print(f"{'=' * 70}")

    print(f"\np4 (DSC Distribution) components:")
    print(f"  Median:   {p4_components['median']:.4f}  (higher is better)")
    print(f"  Variance: {p4_components['variance']:.6f}  (lower is better)")
    print(
        f"  Skewness: {p4_components['skewness']:.4f}  (more negative is better)")
    print(f"  Formula: p4 = 0.6*r_median + 0.25*r_variance + 0.15*r_skewness")

    print(f"\np3 (HD Distribution) components:")
    print(f"  Median:   {p3_components['median']:.2f}  (lower is better)")
    print(f"  Variance: {p3_components['variance']:.2f}  (lower is better)")
    print(
        f"  Skewness: {p3_components['skewness']:.4f}  (more positive is better)")
    print(f"  Formula: p3 = 0.6*r_median + 0.25*r_variance + 0.15*r_skewness")

    print(f"\n{'=' * 70}")
    print(f"Primary Score (Mean DSC): {mean_dice:.4f}")
    print(f"{'=' * 70}")

    # Note about full challenge ranking
    print(f"\nNote: The full SEG.A. challenge uses a comprehensive ranking system:")
    print(f"  - p1: Uniformity of first-order Sobol' indices (p1 = 1 - Σ|Si^(1) - 1/M|)")
    print(f"  - p2: Interaction between image variations (p2 = Σ(Si^(T) - Si^(1)))")
    print(f"  - p3: HD distribution ranking (0.6*r_median + 0.25*r_var + 0.15*r_skew)")
    print(f"  - p4: DSC distribution ranking (0.6*r_median + 0.25*r_var + 0.15*r_skew)")
    print(f"  - Final rank: r_fin = (r1 + r2)/6 + (r3 + r4)/3")
    print(f"\n  This implementation computes p3 and p4 components on the base test set.")
    print(f"  Image variations (rotation, blur, intensity, noise) are not included.")

    # Build results dictionary
    results_dict = {
        'hd_50pc': hd_stats['median'],
        'dsc_50pc': dice_stats['median'],
    }

    # Calculate position and mean position if leaderboard exists
    mean_position_value = None
    leaderboard_path = Path(__file__).parent / 'leaderboard.csv'

    if leaderboard_path.exists():
        try:
            leaderboard_df = pd.read_csv(leaderboard_path)
            submission_dict = {
                "hd_50pc": results_dict["hd_50pc"],
                "dsc_50pc": results_dict["dsc_50pc"],
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
        # Fallback: use negative mean_dice for ranking (higher dice = lower/better rank)
        results_dict['overall'] = -mean_dice

    return results_dict