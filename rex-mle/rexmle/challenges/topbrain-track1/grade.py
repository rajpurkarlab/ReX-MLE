"""
Grading script for TopBrain Track 1 - CTA Multiclass Brain Vessel Segmentation

Implements comprehensive topology-aware metrics including:
1. Class-average Dice similarity coefficient
2. Class-average centerline Dice (clDice)
3. Class-average Betti number error (B0) - connected components
4. Class-average Hausdorff Distance 95th percentile (HD95)
5. Class-average invalid neighbors error
6. Average F1 score for detection of "side road" vessels

Based on TopBrain_Eval_Metrics reference implementation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from collections import defaultdict

import SimpleITK as sitk
from skimage.morphology import skeletonize
from skimage import measure
from monai.metrics.utils import get_mask_edges, get_surface_distance

from rexmle.grade_helpers import InvalidSubmissionError

# Metric directions for position calculation
METRIC_DIRECTIONS = {
    'dice': 'higher_better',
    'cldice': 'higher_better',
    'b0_error': 'lower_better',
    'hd95': 'lower_better',
    'neighbor_error': 'lower_better',
    'f1_side_road': 'higher_better',
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
# Constants for CTA Track 1
# ===========================

# CTA label map (40 vessel classes + background)
MUL_CLASS_LABEL_MAP_CT = {
    "0": "Background",
    "1": "BA",
    "2": "R-P1P2",
    "3": "L-P1P2",
    "4": "R-ICA",
    "5": "R-M1",
    "6": "L-ICA",
    "7": "L-M1",
    "8": "R-Pcom",
    "9": "L-Pcom",
    "10": "Acom",
    "11": "R-A1A2",
    "12": "L-A1A2",
    "13": "R-A3",
    "14": "L-A3",
    "15": "3rd-A2",
    "16": "3rd-A3",
    "17": "R-M2",
    "18": "R-M3",
    "19": "L-M2",
    "20": "L-M3",
    "21": "R-P3P4",
    "22": "L-P3P4",
    "23": "R-VA",
    "24": "L-VA",
    "25": "R-SCA",
    "26": "L-SCA",
    "27": "R-AICA",
    "28": "L-AICA",
    "29": "R-PICA",
    "30": "L-PICA",
    "31": "R-AChA",
    "32": "L-AChA",
    "33": "R-OA",
    "34": "L-OA",
    "35": "VoG",
    "36": "StS",
    "37": "ICVs",
    "38": "R-BVR",
    "39": "L-BVR",
    "40": "SSS",
}

# "Side road" vessels for CTA
SIDEROAD_COMPONENT_LABELS_CT = (
    8, 9, 10, 15, 16,   # R-Pcom, L-Pcom, Acom, 3rd-A2, 3rd-A3
    25, 26,             # R-SCA, L-SCA
    27, 28,             # R-AICA, L-AICA
    29, 30,             # R-PICA, L-PICA
    31, 32,             # R-AChA, L-AChA
    33, 34,             # R-OA, L-OA
    37, 38, 39          # ICVs, R-BVR, L-BVR (CT only)
)

# IoU threshold for detection (lenient threshold)
IOU_THRESHOLD = 0.25

# HD95 upper bound (maximum distance in human head ~290mm)
HD95_UPPER_BOUND = 290


# ===========================
# Utility Functions
# ===========================

def convert_multiclass_to_binary(array: np.ndarray) -> np.ndarray:
    """Merge all non-background labels into binary class."""
    return np.where(array > 0, True, False)


def extract_labels(*arrays: np.ndarray) -> list:
    """Extract unique labels from multiple arrays, excluding background (0)."""
    labels = set()
    for arr in arrays:
        labels.update(np.unique(arr))
    labels.discard(0)
    return sorted(list(labels))


def pad_sitk_image(img: sitk.Image) -> sitk.Image:
    """Pad image to handle completely filled images for distance map."""
    return sitk.ConstantPad(img, [1]*img.GetDimension(), [1]*img.GetDimension(), 0)


def get_neighbor_per_mask(mask_img: sitk.Image) -> dict:
    """
    Get neighbors for each label in the mask.
    Neighborhood is defined by 26-connectivity (3x3x3 cube).
    """
    shifts = [
        [i, j, k]
        for i in [-1, 0, 1]
        for j in [-1, 0, 1]
        for k in [-1, 0, 1]
        if not (i == 0 and j == 0 and k == 0)
    ]

    # SimpleITK ordering is (z,y,x), reorder to (x,y,z)
    mask_arr = sitk.GetArrayFromImage(mask_img).transpose((2, 1, 0)).astype(np.uint8)
    label_array = np.pad(mask_arr, pad_width=1)

    neighbors_dict = defaultdict(set)

    for shift in shifts:
        shifted_array = np.roll(label_array, shift, axis=(0, 1, 2))
        boundary_mask = (label_array != shifted_array) & (label_array != 0)
        boundary_labels = label_array[boundary_mask]
        neighbor_labels = shifted_array[boundary_mask]

        for lab, neigh_lab in zip(boundary_labels, neighbor_labels):
            lab, neigh_lab = int(lab), int(neigh_lab)
            if lab not in neighbors_dict:
                neighbors_dict[lab] = set()
            if neigh_lab != 0:
                neighbors_dict[lab].add(neigh_lab)
                neighbors_dict[neigh_lab].add(lab)

    # Convert to dict with sorted lists
    return {str(k): sorted(v) for k, v in dict(neighbors_dict).items()}


# ===========================
# Metric 1: Dice Coefficient
# ===========================

def dice_coefficient_single_label(gt: sitk.Image, pred: sitk.Image, label: int) -> float:
    """Compute Dice similarity coefficient for a single label."""
    gt_label_arr = sitk.GetArrayFromImage(gt == label)
    pred_label_arr = sitk.GetArrayFromImage(pred == label)

    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        return 0.0

    pred.CopyInformation(gt)

    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.Execute(gt, pred)
    return float(overlap_measures.GetDiceCoefficient(int(label)))


# ===========================
# Metric 2: Centerline Dice (clDice)
# ===========================

def cl_score(s_skeleton: np.ndarray, v_image: np.ndarray) -> float:
    """Compute skeleton volume overlap."""
    if np.sum(s_skeleton) == 0:
        return 0.0
    return float(np.sum(s_skeleton * v_image) / np.sum(s_skeleton))


def clDice_single_label(gt: sitk.Image, pred: sitk.Image, label: int) -> float:
    """Compute centerline Dice for a single label."""
    # Get arrays and transpose from (z,y,x) to (x,y,z)
    gt_label_arr = sitk.GetArrayFromImage(gt == label).transpose((2, 1, 0)).astype(np.uint8)
    pred_label_arr = sitk.GetArrayFromImage(pred == label).transpose((2, 1, 0)).astype(np.uint8)

    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        return 0.0

    # Convert to binary
    pred_mask = convert_multiclass_to_binary(pred_label_arr)
    gt_mask = convert_multiclass_to_binary(gt_label_arr)

    # Compute topology precision and sensitivity
    tprec = cl_score(s_skeleton=skeletonize(pred_mask), v_image=gt_mask)
    tsens = cl_score(s_skeleton=skeletonize(gt_mask), v_image=pred_mask)

    if (tprec + tsens) == 0:
        return 0.0

    return float(2 * tprec * tsens / (tprec + tsens))


# ===========================
# Metric 3: Betti Number Error (B0)
# ===========================

def connected_components(img: np.ndarray) -> int:
    """Calculate B0 Betti number (number of connected components)."""
    assert img.ndim == 3, "Expected 3D input"
    assert np.all((img == 0) | (img == 1)), "Expected binary input"

    # 26-connectivity for foreground
    b0_labels, b0 = measure.label(img, return_num=True, connectivity=3)
    return int(b0)


def betti_number_error_single_label(gt: sitk.Image, pred: sitk.Image, label: int) -> int:
    """Compute Betti number error for a single label."""
    gt_label_arr = sitk.GetArrayFromImage(gt == label).astype(np.uint8)
    pred_label_arr = sitk.GetArrayFromImage(pred == label).astype(np.uint8)

    gt_b0 = 0 if not np.any(gt_label_arr) else connected_components(gt_label_arr)
    pred_b0 = 0 if not np.any(pred_label_arr) else connected_components(pred_label_arr)

    return abs(pred_b0 - gt_b0)



def hd95_single_label(gt: sitk.Image, pred: sitk.Image, label: int) -> float:
    """Compute Hausdorff Distance 95th percentile for a single label using MONAI."""
    gt_label_arr = sitk.GetArrayFromImage(gt == label).astype(np.uint8)
    pred_label_arr = sitk.GetArrayFromImage(pred == label).astype(np.uint8)

    # Handle empty masks
    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        return HD95_UPPER_BOUND

    # Get spacing from the image
    spacing = gt.GetSpacing()[::-1]  # Convert from (x,y,z) to (z,y,x) for numpy array ordering

    # Use MONAI's faster implementation
    edges_pred, edges_gt = get_mask_edges(pred_label_arr, gt_label_arr)

    # get_surface_distance returns all surface distances
    surface_distances = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean", spacing=spacing)

    # If no surface distances, return upper bound
    if surface_distances.shape == (0,):
        return HD95_UPPER_BOUND

    # Compute 95th percentile
    hd95 = float(np.percentile(surface_distances, 95))

    return hd95


# ===========================
# Metric 5: Invalid Neighbors
# ===========================

def invalid_neighbors_error(gt_neighbors: dict, pred_neighbors: dict, label: int) -> int:
    """Count invalid neighbors for a single label."""
    gt_neigh = set(gt_neighbors.get(str(label), []))
    pred_neigh = set(pred_neighbors.get(str(label), []))

    # Invalid neighbors are those in pred but not in gt
    invalid = pred_neigh - gt_neigh
    return len(invalid)


# ===========================
# Metric 6: Side Road Detection
# ===========================

def iou_single_label(gt: sitk.Image, pred: sitk.Image, label: int) -> float:
    """Compute IoU (Jaccard coefficient) for a single label."""
    gt_label_arr = sitk.GetArrayFromImage(gt == label)
    pred_label_arr = sitk.GetArrayFromImage(pred == label)

    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        return 0.0

    pred.CopyInformation(gt)

    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.Execute(gt, pred)
    return float(overlap_measures.GetJaccardCoefficient(int(label)))


def detection_sideroad_labels(gt: sitk.Image, pred: sitk.Image) -> dict:
    """
    Detect side road vessels.
    Returns dict with TP/TN/FP/FN for each side road label.
    """
    gt_array = sitk.GetArrayFromImage(gt)
    pred_array = sitk.GetArrayFromImage(pred)

    gt_labels = set(extract_labels(gt_array))
    pred_labels = set(extract_labels(pred_array))

    detection_dict = {}

    for label in SIDEROAD_COMPONENT_LABELS_CT:
        if label in gt_labels:
            iou_score = iou_single_label(gt, pred, label)
            detection = "TP" if iou_score >= IOU_THRESHOLD else "FN"
        else:
            detection = "FP" if label in pred_labels else "TN"

        detection_dict[str(label)] = {
            "label": MUL_CLASS_LABEL_MAP_CT[str(label)],
            "Detection": detection,
        }

    return detection_dict


def aggregate_detection_dicts(all_detection_dicts: list) -> float:
    """Aggregate detection dicts and compute average F1 score."""
    detection_counts = {}

    for label in SIDEROAD_COMPONENT_LABELS_CT:
        detection_counts[str(label)] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for detection_dict in all_detection_dicts:
        for label, value in detection_dict.items():
            detection_counts[label][value["Detection"]] += 1

    f1_scores = []
    for label, stats in detection_counts.items():
        tp = stats["TP"]
        fp = stats["FP"]
        fn = stats["FN"]

        if (tp + fp + fn) == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * tp / ((2 * tp) + fp + fn)

        f1_scores.append(f1_score)

    return float(np.mean(f1_scores))


# ===========================
# Main Grading Function
# ===========================

def compute_all_metrics_for_case(gt_sitk: sitk.Image, pred_sitk: sitk.Image,
                                  valid_neighbors: dict) -> dict:
    """Compute all TopBrain metrics for a single case."""
    # Cast to same type
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkUInt8)
    gt_sitk = caster.Execute(gt_sitk)
    pred_sitk = caster.Execute(pred_sitk)

    # Ensure same metadata
    pred_sitk.CopyInformation(gt_sitk)

    # Get arrays
    gt_arr = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)
    pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(np.uint8)

    # Get all present labels (excluding background)
    labels = extract_labels(gt_arr, pred_arr)

    if len(labels) == 0:
        return {
            "dice": 0.0,
            "cldice": 0.0,
            "b0_error": 0.0,
            "hd95": 0.0,
            "invalid_neighbors": 0.0,
            "detection_dict": {}
        }

    # Compute metrics for each class
    dice_scores = []
    cldice_scores = []
    b0_errors = []
    hd95_scores = []
    invalid_nb_errors = []

    # Get pred neighbors for invalid neighbor metric
    pred_neighbors = get_neighbor_per_mask(pred_sitk)

    for label in labels:
        # Dice
        dice_scores.append(dice_coefficient_single_label(gt_sitk, pred_sitk, label))

        # clDice
        cldice_scores.append(clDice_single_label(gt_sitk, pred_sitk, label))

        # B0 error
        b0_errors.append(betti_number_error_single_label(gt_sitk, pred_sitk, label))

        # HD95
        hd95_scores.append(hd95_single_label(gt_sitk, pred_sitk, label))

        # Invalid neighbors
        invalid_nb_errors.append(invalid_neighbors_error(valid_neighbors, pred_neighbors, label))

    # Detection for side road vessels
    detection_dict = detection_sideroad_labels(gt_sitk, pred_sitk)

    return {
        "dice": float(np.mean(dice_scores)),
        "cldice": float(np.mean(cldice_scores)),
        "b0_error": float(np.mean(b0_errors)),
        "hd95": float(np.mean(hd95_scores)),
        "invalid_neighbors": float(np.mean(invalid_nb_errors)),
        "detection_dict": detection_dict
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame,
          submission_dir: Optional[Path] = None, answers_dir: Optional[Path] = None) -> dict:
    """
    Grade TopBrain Track 1 (CTA) submission using comprehensive TopBrain metrics.

    Returns dict with all metrics:
    - dice, cldice, b0_error, hd95, invalid_neighbors, f1_sideroad
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

    # Load valid neighbors for CTA
    valid_neighbors_path = Path(__file__).parent / "valid_neighbors_ct_all.json"
    with open(valid_neighbors_path, 'r') as f:
        valid_neighbors = json.load(f)

    # Compute metrics for each case
    all_dice = []
    all_cldice = []
    all_b0_error = []
    all_hd95 = []
    all_invalid_neighbors = []
    all_detection_dicts = []

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
            # Load images using SimpleITK
            gt_sitk = sitk.ReadImage(str(gt_mask_path))
            pred_sitk = sitk.ReadImage(str(pred_mask_path))

            # Resize prediction if shapes don't match (use nearest neighbor for segmentation masks)
            if gt_sitk.GetSize() != pred_sitk.GetSize():
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Shape mismatch for {image_id}: "
                    f"predicted {pred_sitk.GetSize()} vs ground truth {gt_sitk.GetSize()}. "
                    f"Resizing prediction to match ground truth."
                )
                # Resample prediction to match ground truth size
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(gt_sitk)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest neighbor for segmentation
                resampler.SetDefaultPixelValue(0)
                pred_sitk = resampler.Execute(pred_sitk)

            # Compute all metrics
            metrics = compute_all_metrics_for_case(gt_sitk, pred_sitk, valid_neighbors)

            all_dice.append(metrics["dice"])
            all_cldice.append(metrics["cldice"])
            all_b0_error.append(metrics["b0_error"])
            all_hd95.append(metrics["hd95"])
            all_invalid_neighbors.append(metrics["invalid_neighbors"])
            all_detection_dicts.append(metrics["detection_dict"])

        except Exception as e:
            raise InvalidSubmissionError(f"Error evaluating {image_id}: {e}")

    # Aggregate metrics
    if len(all_dice) == 0:
        raise InvalidSubmissionError("No metrics computed")

    mean_dice = float(np.mean(all_dice))
    mean_cldice = float(np.mean(all_cldice))
    mean_b0_error = float(np.mean(all_b0_error))
    mean_hd95 = float(np.mean(all_hd95))
    mean_invalid_neighbors = float(np.mean(all_invalid_neighbors))

    # Aggregate detection dicts for F1 score
    f1_sideroad = aggregate_detection_dicts(all_detection_dicts)

    total_cases = len(all_dice)

    print(f"\n=== TopBrain Track 1 (CTA) Results ===")
    print(f"Cases evaluated: {total_cases}")
    print(f"Class-avg Dice: {mean_dice:.4f}")
    print(f"Class-avg clDice: {mean_cldice:.4f}")
    print(f"Class-avg B0 error: {mean_b0_error:.4f}")
    print(f"Class-avg HD95: {mean_hd95:.4f}")
    print(f"Class-avg Invalid Neighbors: {mean_invalid_neighbors:.4f}")
    print(f"F1 Side Road Vessels: {f1_sideroad:.4f}")

    # Build results dictionary
    results_dict = {
        "dice": mean_dice,
        "cldice": mean_cldice,
        "b0_error": mean_b0_error,
        "hd95": mean_hd95,
        "neighbor_error": mean_invalid_neighbors,
        "f1_side_road": f1_sideroad,
    }

    # Calculate position and mean position if leaderboard exists
    mean_position_value = None
    leaderboard_path = Path(__file__).parent / 'leaderboard.csv'

    if leaderboard_path.exists():
        try:
            leaderboard_df = pd.read_csv(leaderboard_path)
            print("leaderboard_df", leaderboard_df)
            submission_dict = {
                "dice": results_dict["dice"],
                "cldice": results_dict["cldice"],
                "b0_error": results_dict["b0_error"],
                "hd95": results_dict["hd95"],
                "neighbor_error": results_dict["neighbor_error"],
                "f1_side_road": results_dict["f1_side_road"],
            }
            positions = calculate_positions_and_mean(submission_dict, leaderboard_df)
            print("positions", positions)
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
        # Fallback: use negative mean_dice for ranking (higher dice = lower/better rank)
        results_dict['overall'] = -results_dict['dice']

    return results_dict
