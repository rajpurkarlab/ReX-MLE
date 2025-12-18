"""
Grading script for TopCoW Track 1 Task 1 - CTA Multi-class Segmentation

Implements comprehensive metrics including:
- dice_per_case: Class-average Dice similarity coefficient
- cldice: Centerline Dice for topology evaluation
- b0_error: Betti number error (connected components)
- hd95: Hausdorff Distance 95th percentile
- f1_grp2: F1 score for Group 2 CoW components detection
- anterior_accuracy: Anterior topology match accuracy
- posterior_accuracy: Posterior topology match accuracy
- anterior_topology: Anterior topology match rate
- posterior_topology: Posterior topology match rate

Based on TopCoW_Eval_Metrics reference implementation.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Optional, Dict, Tuple
import SimpleITK as sitk
from skimage.morphology import skeletonize
from skimage import measure
from sklearn.metrics import balanced_accuracy_score
from monai.metrics.utils import get_mask_edges, get_surface_distance

from rexmle.grade_helpers import InvalidSubmissionError

# Metric directions for position calculation
METRIC_DIRECTIONS = {
    'dice': 'higher_better',
    'cldice': 'higher_better',
    'b0_error': 'lower_better',
    'hd95': 'lower_better',
    'f1_grp2': 'higher_better',
    'anterior_graph_acc': 'higher_better',
    'posterior_graph_acc': 'higher_better',
    'anterior_topology': 'higher_better',
    'posterior_topology': 'higher_better',
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

# Constants from TopCoW_Eval_Metrics
GROUP2_COW_COMPONENTS_LABELS = (8, 9, 10, 15)  # R-Pcom, L-Pcom, Acom, 3rd-A2
IOU_THRESHOLD = 0.25
HD95_UPPER_BOUND = 90  # mm
ANTERIOR_LABELS = (10, 11, 12, 15)  # Acom, R-ACA, L-ACA, 3rd-A2
POSTERIOR_LABELS = (2, 3, 8, 9)  # R-PCA, L-PCA, R-Pcom, L-Pcom

MUL_CLASS_LABEL_MAP = {
    0: "Background",
    1: "BA",
    2: "R-PCA",
    3: "L-PCA",
    4: "R-ICA",
    5: "R-MCA",
    6: "L-ICA",
    7: "L-MCA",
    8: "R-Pcom",
    9: "L-Pcom",
    10: "Acom",
    11: "R-ACA",
    12: "L-ACA",
    15: "3rd-A2",
}


def compute_dice_per_class(pred_mask: np.ndarray, gt_mask: np.ndarray, class_id: int) -> float:
    """Compute Dice similarity coefficient for a specific class."""
    pred_binary = (pred_mask == class_id).astype(np.uint8)
    gt_binary = (gt_mask == class_id).astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()

    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    elif gt_sum == 0:
        return 0.0
    else:
        return float(2.0 * intersection / (pred_sum + gt_sum + 1e-7))


def compute_class_average_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute class-average Dice similarity coefficient."""
    classes = sorted(np.unique(gt_mask))
    if 0 in classes:
        classes.remove(0)

    if len(classes) == 0:
        return 1.0

    dice_scores = []
    for class_id in classes:
        dice = compute_dice_per_class(pred_mask, gt_mask, class_id)
        dice_scores.append(dice)

    return float(np.mean(dice_scores))


def convert_multiclass_to_binary(array: np.ndarray) -> np.ndarray:
    """Merge all non-background labels into binary class."""
    return np.where(array > 0, True, False)


def cl_score(s_skeleton: np.ndarray, v_image: np.ndarray) -> float:
    """Compute skeleton volume overlap."""
    if np.sum(s_skeleton) == 0:
        return 0
    return float(np.sum(s_skeleton * v_image) / np.sum(s_skeleton))


def compute_cldice(v_p_pred: np.ndarray, v_l_gt: np.ndarray) -> float:
    """Compute centerline Dice metric."""
    pred_mask = convert_multiclass_to_binary(v_p_pred)
    gt_mask = convert_multiclass_to_binary(v_l_gt)

    if len(pred_mask.shape) != 3:
        return 0.0

    tprec = cl_score(s_skeleton=skeletonize(pred_mask), v_image=gt_mask)
    tsens = cl_score(s_skeleton=skeletonize(gt_mask), v_image=pred_mask)

    if (tprec + tsens) == 0:
        return 0.0

    return float(2 * tprec * tsens / (tprec + tsens))


def connected_components(img: np.ndarray) -> tuple:
    """Identify connected components and calculate B0 Betti number."""
    assert img.ndim == 3, "betti_number expects a 3D input"
    N26 = 3  # full connectivity

    b0_labels, b0 = measure.label(img, return_num=True, connectivity=N26)
    props = measure.regionprops(b0_labels)
    sizes = [obj.area for obj in props]
    sizes.sort()

    return int(b0), props, sizes


def compute_b0_error_per_class(gt_mask: np.ndarray, pred_mask: np.ndarray, class_id: int) -> int:
    """Compute Betti number error for a specific class."""
    gt_binary = (gt_mask == class_id).astype(np.uint8)
    pred_binary = (pred_mask == class_id).astype(np.uint8)

    gt_b0 = 0 if not np.any(gt_binary) else connected_components(gt_binary)[0]
    pred_b0 = 0 if not np.any(pred_binary) else connected_components(pred_binary)[0]

    return abs(pred_b0 - gt_b0)


def compute_class_average_b0_error(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Compute class-average Betti number error."""
    classes = sorted(np.unique(gt_mask))
    if 0 in classes:
        classes.remove(0)

    if len(classes) == 0:
        return 0.0

    b0_errors = []
    for class_id in classes:
        b0_err = compute_b0_error_per_class(gt_mask, pred_mask, class_id)
        b0_errors.append(b0_err)

    return float(np.mean(b0_errors))


def compute_hd95_per_class(gt_sitk: sitk.Image, pred_sitk: sitk.Image, class_id: int) -> float:
    """Compute HD95 for a specific class using MONAI utilities."""
    gt_label_img = gt_sitk == class_id
    pred_label_img = pred_sitk == class_id

    gt_label_arr = sitk.GetArrayFromImage(gt_label_img).astype(np.uint8)
    pred_label_arr = sitk.GetArrayFromImage(pred_label_img).astype(np.uint8)

    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        return HD95_UPPER_BOUND

    # Get spacing from SimpleITK image (in z, y, x order for numpy array)
    spacing = gt_sitk.GetSpacing()[::-1]  # Convert from x,y,z to z,y,x

    # Use MONAI utilities to compute surface distances
    edges_pred, edges_gt = get_mask_edges(pred_label_arr, gt_label_arr)

    # get_surface_distance computes distances from edges_pred to edges_gt and vice versa
    surface_distances = get_surface_distance(
        edges_pred, edges_gt,
        distance_metric="euclidean",
        spacing=spacing
    )

    # If surface_distances is empty, return upper bound
    if surface_distances.shape == (0,):
        return HD95_UPPER_BOUND

    # Compute 95th percentile (HD95)
    hd95 = float(np.percentile(surface_distances, 95))

    return hd95


def compute_class_average_hd95(gt_sitk: sitk.Image, pred_sitk: sitk.Image) -> float:
    """Compute class-average HD95."""
    gt_arr = sitk.GetArrayFromImage(gt_sitk)
    classes = sorted(np.unique(gt_arr))
    if 0 in classes:
        classes.remove(0)

    if len(classes) == 0:
        return 0.0

    hd95_scores = []
    for class_id in classes:
        hd95 = compute_hd95_per_class(gt_sitk, pred_sitk, class_id)
        hd95_scores.append(hd95)

    return float(np.mean(hd95_scores))


def compute_iou_single_label(gt_sitk: sitk.Image, pred_sitk: sitk.Image, label: int) -> float:
    """Compute IoU for a single label."""
    gt_label_arr = sitk.GetArrayFromImage(gt_sitk == label)
    pred_label_arr = sitk.GetArrayFromImage(pred_sitk == label)

    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        return 0.0

    pred_sitk.CopyInformation(gt_sitk)

    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(gt_sitk, pred_sitk)
    iou_score = overlap_measures.GetJaccardCoefficient(label)
    return float(iou_score)


def compute_detection_dict(gt_sitk: sitk.Image, pred_sitk: sitk.Image) -> dict:
    """Compute detection dict for Group 2 CoW components."""
    gt_array = sitk.GetArrayFromImage(gt_sitk)
    pred_array = sitk.GetArrayFromImage(pred_sitk)

    gt_labels = set(np.unique(gt_array)) - {0}
    pred_labels = set(np.unique(pred_array)) - {0}

    detection_dict = {}

    for label in GROUP2_COW_COMPONENTS_LABELS:
        if label in gt_labels:
            iou_score = compute_iou_single_label(gt_sitk, pred_sitk, label)
            detection = "TP" if iou_score >= IOU_THRESHOLD else "FN"
        else:
            detection = "FP" if label in pred_labels else "TN"

        detection_dict[str(label)] = {
            "label": MUL_CLASS_LABEL_MAP[label],
            "Detection": detection,
        }

    return detection_dict


def compute_f1_from_detection_dicts(all_detection_dicts: list) -> float:
    """Aggregate detection dicts and compute average F1 score."""
    detection_counts = {}

    for label in GROUP2_COW_COMPONENTS_LABELS:
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


def compute_topology_dict(gt_sitk: sitk.Image, pred_sitk: sitk.Image) -> dict:
    """Compute simplified topology matching dict."""
    # This is a simplified version - full topology matching is complex
    # For now, we'll do basic detection-based topology matching
    gt_array = sitk.GetArrayFromImage(gt_sitk)
    pred_array = sitk.GetArrayFromImage(pred_sitk)

    gt_labels = set(np.unique(gt_array)) - {0}
    pred_labels = set(np.unique(pred_array)) - {0}

    def check_topology_match(labels):
        matches = []
        for label in labels:
            if label in gt_labels:
                iou = compute_iou_single_label(gt_sitk, pred_sitk, label)
                match = iou >= IOU_THRESHOLD
            else:
                match = label not in pred_labels
            matches.append(match)
        return all(matches)

    anterior_match = check_topology_match(ANTERIOR_LABELS)
    posterior_match = check_topology_match(POSTERIOR_LABELS)

    return {
        "match_verdict": {
            "anterior": anterior_match,
            "posterior": posterior_match,
        }
    }


def compute_topology_accuracy_from_dicts(all_topo_dicts: list) -> tuple:
    """Compute topology accuracy metrics from topology dicts."""
    # For simplified topology matching, we use the match verdicts directly
    anterior_matches = [td["match_verdict"]["anterior"] for td in all_topo_dicts]
    posterior_matches = [td["match_verdict"]["posterior"] for td in all_topo_dicts]

    # Compute balanced accuracy (treating True/False as classes)
    # Create simple y_true and y_pred where all ground truth is "correct"
    n = len(anterior_matches)

    # For anterior
    ant_y_true = ["correct"] * n
    ant_y_pred = ["correct" if match else "incorrect" for match in anterior_matches]
    anterior_accuracy = float(balanced_accuracy_score(ant_y_true, ant_y_pred))

    # For posterior
    pos_y_true = ["correct"] * n
    pos_y_pred = ["correct" if match else "incorrect" for match in posterior_matches]
    posterior_accuracy = float(balanced_accuracy_score(pos_y_true, pos_y_pred))

    # Topology match rates
    anterior_topology = float(sum(anterior_matches) / n)
    posterior_topology = float(sum(posterior_matches) / n)

    return anterior_accuracy, posterior_accuracy, anterior_topology, posterior_topology


def compute_topcow_metrics(gt_sitk: sitk.Image, pred_sitk: sitk.Image) -> dict:
    """Compute all TopCoW metrics for a single case."""
    # Cast to the same type
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkUInt8)
    caster.SetNumberOfThreads(1)
    gt_sitk = caster.Execute(gt_sitk)
    pred_sitk = caster.Execute(pred_sitk)

    # Make sure they have the same metadata
    pred_sitk.CopyInformation(gt_sitk)

    # Get arrays
    gt_arr = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)
    pred_arr = sitk.GetArrayFromImage(pred_sitk).astype(np.uint8)

    metrics = {}

    # (1) Dice coefficient per class
    metrics["dice_per_case"] = compute_class_average_dice(pred_arr, gt_arr)

    # (2) Centerline Dice (clDice)
    metrics["cldice"] = compute_cldice(pred_arr, gt_arr)

    # (3) Betti number error (B0)
    metrics["b0_error"] = compute_class_average_b0_error(gt_arr, pred_arr)

    # (4) Hausdorff Distance 95th percentile
    metrics["hd95"] = compute_class_average_hd95(gt_sitk, pred_sitk)

    # (5) Detection for Group 2 CoW components
    metrics["detection_dict"] = compute_detection_dict(gt_sitk, pred_sitk)

    # (6) Topology matching
    metrics["topo_dict"] = compute_topology_dict(gt_sitk, pred_sitk)

    return metrics


def grade(submission: pd.DataFrame, answers: pd.DataFrame, submission_dir: Optional[Path] = None, answers_dir: Optional[Path] = None) -> dict:
    """
    Grade TopCoW Track 1 Task 1 submission using comprehensive TopCoW metrics.

    Returns all metrics from leaderboard:
    - dice_per_case, cldice, b0_error, hd95, f1_grp2,
    - anterior_accuracy, posterior_accuracy, anterior_topology, posterior_topology
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

    # Compute metrics for each case
    all_dice_scores = []
    all_cldice_scores = []
    all_b0_errors = []
    all_hd95_scores = []
    all_detection_dicts = []
    all_topo_dicts = []

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

            # Compute all TopCoW metrics
            metrics = compute_topcow_metrics(gt_sitk, pred_sitk)

            all_dice_scores.append(metrics["dice_per_case"])
            all_cldice_scores.append(metrics["cldice"])
            all_b0_errors.append(metrics["b0_error"])
            all_hd95_scores.append(metrics["hd95"])
            all_detection_dicts.append(metrics["detection_dict"])
            all_topo_dicts.append(metrics["topo_dict"])

        except Exception as e:
            raise InvalidSubmissionError(f"Error evaluating {image_id}: {e}")

    # Aggregate metrics
    if len(all_dice_scores) == 0:
        raise InvalidSubmissionError("No metrics computed")

    # Mean metrics
    mean_dice = float(np.mean(all_dice_scores))
    mean_cldice = float(np.mean(all_cldice_scores))
    mean_b0_error = float(np.mean(all_b0_errors))
    mean_hd95 = float(np.mean(all_hd95_scores))

    # Aggregate detection dicts for F1 score
    f1_grp2 = compute_f1_from_detection_dicts(all_detection_dicts)

    # Aggregate topology dicts
    anterior_accuracy, posterior_accuracy, anterior_topology, posterior_topology = \
        compute_topology_accuracy_from_dicts(all_topo_dicts)

    total_cases = len(all_dice_scores)

    print(f"\n=== TopCoW Track 1 Task 1 Results ===")
    print(f"Cases evaluated: {total_cases}")
    print(f"Dice per case: {mean_dice:.4f}")
    print(f"clDice: {mean_cldice:.4f}")
    print(f"B0 error: {mean_b0_error:.4f}")
    print(f"HD95: {mean_hd95:.4f}")
    print(f"F1 Group 2: {f1_grp2:.4f}")
    print(f"Anterior accuracy: {anterior_accuracy:.4f}")
    print(f"Posterior accuracy: {posterior_accuracy:.4f}")
    print(f"Anterior topology: {anterior_topology:.4f}")
    print(f"Posterior topology: {posterior_topology:.4f}")

    # Build results dictionary
    results_dict = {
        "dice": mean_dice,
        "cldice": mean_cldice,
        "b0_error": mean_b0_error,
        "hd95": mean_hd95,
        "f1_grp2": f1_grp2,
        "anterior_graph_acc": anterior_accuracy,
        "posterior_graph_acc": posterior_accuracy,
        "anterior_topology": anterior_topology,
        "posterior_topology": posterior_topology,
    }

    # Calculate position and mean position if leaderboard exists
    mean_position_value = None
    leaderboard_path = Path(__file__).parent / 'leaderboard.csv'

    if leaderboard_path.exists():
        try:
            leaderboard_df = pd.read_csv(leaderboard_path)
            submission_dict = {k: results_dict[k] for k in results_dict.keys()}
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
        # Fallback: use negative mean_dice for ranking (higher dice = lower/better rank)
        results_dict['overall'] = -results_dict['dice']

    return results_dict
