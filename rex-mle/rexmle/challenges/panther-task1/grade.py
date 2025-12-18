"""
Grading script for PANTHER Task 1 - Pancreatic Tumor Segmentation on Diagnostic MRI

Uses the official surface-distance library from DeepMind for metric computation,
matching the PANTHER challenge evaluation code.

Evaluation Metrics:
1. Dice Similarity Coefficient (DSC) - Primary metric
2. 5mm Surface Dice (MSD)
3. Mean Average Surface Distance (MASD)
4. Hausdorff Distance 95% (HD95)
5. RMSE on Tumor Burden
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict

try:
    from surface_distance import metrics as surface_metrics
except ImportError:
    raise ImportError(
        "surface-distance library is required for PANTHER evaluation. "
        "Install with: pip install surface-distance"
    )

from rexmle.grade_helpers import InvalidSubmissionError
from rexmle.medical_utils import load_medical_image


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple) -> Dict[str, float]:
    """
    Compute all PANTHER evaluation metrics using official surface-distance library.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        spacing: Voxel spacing (x, y, z) in mm

    Returns:
        Dictionary of metrics
    """
    # Convert to boolean as required by surface-distance library
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    # Handle uniform predictions (all zeros or all ones)
    if np.all(pred_bool == 0) or np.all(pred_bool == 1):
        max_distance = np.linalg.norm(np.array(gt_bool.shape) * np.array(spacing))
        voxel_volume = np.prod(spacing)
        gt_volume = np.sum(gt_bool) * voxel_volume

        return {
            'dice': 0.0,
            'surface_dice_5mm': 0.0,
            'hd95': max_distance,
            'masd': max_distance,
            'tumor_burden_rmse': gt_volume  # RMSE when pred_volume = 0
        }

    # Compute surface distances
    surface_distances = surface_metrics.compute_surface_distances(
        gt_bool, pred_bool, spacing_mm=spacing
    )

    # Compute metrics
    dice = surface_metrics.compute_dice_coefficient(gt_bool, pred_bool)
    surf_dice = surface_metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=5)
    hd95 = surface_metrics.compute_robust_hausdorff(surface_distances, percent=95)

    avg_gt_to_pred, avg_pred_to_gt = surface_metrics.compute_average_surface_distance(surface_distances)
    masd = (avg_gt_to_pred + avg_pred_to_gt) / 2.0

    # Compute tumor burden RMSE
    voxel_volume = np.prod(spacing)
    gt_volume = np.sum(gt_bool) * voxel_volume
    pred_volume = np.sum(pred_bool) * voxel_volume
    tumor_burden_rmse = np.sqrt((pred_volume - gt_volume) ** 2)

    return {
        'dice': float(dice),
        'surface_dice_5mm': float(surf_dice),
        'hd95': float(hd95),
        'masd': float(masd),
        'tumor_burden_rmse': float(tumor_burden_rmse)
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame, submission_dir: Optional[Path] = None, answers_dir: Optional[Path] = None) -> float:
    """
    Grade a PANTHER Task 1 submission.

    Args:
        submission: Submission DataFrame with columns: image_id, predicted_mask_path
        answers: Ground truth DataFrame with columns: image_id, image_path, label_path
        submission_dir: Directory containing submission files
        answers_dir: Directory containing ground truth files

    Returns:
        Primary score (Dice Similarity Coefficient averaged across all cases)
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

    # Compute metrics for each case
    all_dice_scores = []
    all_metrics = []

    for idx, row in submission.iterrows():
        image_id = row['image_id']
        pred_mask_path = submission_dir / row['predicted_mask_path']

        # Find ground truth
        gt_row = answers[answers['image_id'] == image_id]
        if gt_row.empty:
            raise InvalidSubmissionError(f"No ground truth found for {image_id}")

        # Ground truth path is relative to answers_dir
        gt_mask_path = answers_dir / gt_row.iloc[0]['label_path']

        # Validate predicted mask exists
        if not pred_mask_path.exists():
            raise InvalidSubmissionError(f"Predicted mask not found: {pred_mask_path}")

        # Load masks
        try:
            pred_data = load_medical_image(pred_mask_path)
            pred_mask = pred_data['image']
            spacing = tuple(pred_data['metadata'].get('spacing', [1.0, 1.0, 1.0]))
        except Exception as e:
            raise InvalidSubmissionError(f"Error loading predicted mask {pred_mask_path}: {e}")

        try:
            gt_data = load_medical_image(gt_mask_path)
            gt_mask = gt_data['image']
            # PANTHER labels: 0=background, 1=tumor, 2=pancreas
            # We only need tumor (value 1)
            gt_mask = (gt_mask == 1).astype(np.uint8)
        except Exception as e:
            raise InvalidSubmissionError(f"Error loading ground truth mask {gt_mask_path}: {e}")

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

        # Ensure prediction is binary or extract tumor if multi-class
        unique_vals = np.unique(pred_mask)
        if not (np.array_equal(unique_vals, [0]) or
                np.array_equal(unique_vals, [0, 1]) or
                np.array_equal(unique_vals, [1])):
            # Check if it's PANTHER format with tumor as value 1
            if 1 in unique_vals and 2 in unique_vals:
                # Extract tumor only (value 1)
                pred_mask = (pred_mask == 1).astype(np.uint8)
            elif len(unique_vals) == 2 and 0 in unique_vals:
                # Convert nonzero to 1
                pred_mask = (pred_mask > 0).astype(np.uint8)
            else:
                raise InvalidSubmissionError(
                    f"Prediction mask for {image_id} is not binary. Unique values: {unique_vals}"
                )

        # Compute metrics
        case_metrics = compute_metrics(pred_mask, gt_mask, spacing=spacing)
        case_metrics['image_id'] = image_id

        all_dice_scores.append(case_metrics['dice'])
        all_metrics.append(case_metrics)

    # Return primary metric (mean Dice)
    mean_dice = np.mean(all_dice_scores)
    msd = np.mean([m['surface_dice_5mm'] for m in all_metrics])
    hd95 = np.mean([m['hd95'] for m in all_metrics])
    masd = np.mean([m['masd'] for m in all_metrics])
    rmse = np.sqrt(np.mean([m['tumor_burden_rmse']**2 for m in all_metrics]))


    # Print summary metrics
    print(f"\n=== PANTHER Task 1 Evaluation Results ===")
    print(f"Cases evaluated: {len(all_metrics)}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean Surface Dice (5mm): {msd:.4f}")
    print(f"Mean HD95: {hd95:.2f} mm")
    print(f"Mean MASD: {masd:.2f} mm")
    print(f"RMSE Tumor Burden: {rmse:.2f} mm³")

    results_dict = {
        "dsc": float(mean_dice),
        "msd": float(msd),
        "hd95": float(hd95),
        "masd": float(masd),
        "rmse": float(rmse)
    }

    # Calculate mean_position from all metric positions
    leaderboard_path = Path(__file__).parent / 'leaderboard.csv'
    mean_position_value = None
    if leaderboard_path.exists():
        leaderboard_df = pd.read_csv(leaderboard_path)
        # Remove position columns from leaderboard for comparison
        metric_cols = [col for col in leaderboard_df.columns
                       if not col.endswith('_position') and col not in ['rank', 'team', 'mean_position']]

        # Calculate positions for this submission
        submission_dict = {k: v for k, v in results_dict.items() if k in metric_cols}

        if submission_dict:
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

    # Use mean_position for overall ranking (lower is better)
    if mean_position_value is not None:
        results_dict['overall'] = mean_position_value
    else:
        # Fallback: use negated primary metric
        results_dict['overall'] = -list(results_dict.values())[0]

    return results_dict




# Metric direction configuration and position calculation
METRIC_DIRECTIONS = {
    "dsc": "higher_better",
    "msd": "higher_better",  # Surface Dice: higher is better (1.0 is perfect)
    "hd95": "lower_better",
    "masd": "lower_better",
    "rmse": "lower_better"
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
