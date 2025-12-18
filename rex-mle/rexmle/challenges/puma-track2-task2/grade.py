"""
Grading script for PUMA Track 2 Task 2 - Nuclei Detection (10 Classes)

Evaluation Metric:
- Macro F1 Score: Based on hit criterion using confidence score and centroid distance

Evaluation Process (from official PUMA evaluation code):
1. Extract nuclei centroids and classes from polygon annotations
2. Match predictions to ground truth within 15-pixel radius:
   - Sort by confidence score (descending) then distance (ascending)
   - Match best prediction to each ground truth
   - One-to-one matching (each prediction and GT can only be matched once)
3. Count TP/FP/FN per class
4. Compute F1 score per class, then average (macro)

Classes (10):
- tumor
- lymphocytes
- plasma_cells
- histiocytes
- melanophages
- neutrophils
- stromal_cells
- epithelium
- endothelium
- apoptotic_cells

Based on official PUMA evaluation code from:
https://github.com/PUMA-Challenge/PUMA-challenge-eval-track2
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from rexmle.grade_helpers import InvalidSubmissionError

# Metric directions for position calculation
METRIC_DIRECTIONS = {
    "f1": "higher_better"
}


def normalize_category_name(raw_name: str) -> Optional[str]:
    """
    Map various raw category names (including nuclei_* variants) to canonical names.
    Returns None if the name cannot be mapped.
    """
    if raw_name is None:
        return None

    name = str(raw_name).strip().lower()

    canonical = {
        "tumor": "tumor",
        "epithelium": "epithelium",
        "endothelium": "endothelium",
        "apoptotic_cells": "apoptotic_cells",
        "lymphocytes": "lymphocytes",
        "plasma_cells": "plasma_cells",
        "histiocytes": "histiocytes",
        "melanophages": "melanophages",
        "neutrophils": "neutrophils",
        "stromal_cells": "stromal_cells",
    }
    if name in canonical:
        return canonical[name]

    if name.startswith("nuclei_"):
        trimmed = name[len("nuclei_"):]
        nuclei_map = {
            "lymphocyte": "lymphocytes",
            "plasma_cell": "plasma_cells",
            "histiocyte": "histiocytes",
            "melanophage": "melanophages",
            "neutrophil": "neutrophils",
            "stroma": "stromal_cells",
            "apoptosis": "apoptotic_cells",
        }
        return nuclei_map.get(trimmed)

    return None


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


def calculate_centroid(points: List[List[float]]) -> np.ndarray:
    """
    Calculate the centroid of a polygon given its points.
    Points should be a list of [x, y] coordinates.
    """
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    return centroid


def extract_features_from_json(json_data: dict, json_name: str) -> List[Dict]:
    """
    Extract nuclei features from JSON data in "Multiple Polygons" format.

    Expected format:
    {
        "polygons": [
            {
                "name": "tumor",
                "path_points": [[x1, y1], [x2, y2], ...],
                "score": 0.95  (optional)
            },
            ...
        ]
    }

    Returns list of features with centroid, category, and score.
    """
    features_list = []

    # Support both "polygons" (official format) and "nuclei" (simplified format)
    polygons = json_data.get('polygons', json_data.get('nuclei', []))

    for polygon_data in polygons:
        # Get category/class name and normalize to canonical label set
        raw_category = polygon_data.get('name', polygon_data.get('class'))
        category = normalize_category_name(raw_category)
        if category is None:
            continue

        # Get confidence score (default to 1.0 if not provided)
        score = polygon_data.get('score', polygon_data.get('confidence', 1.0))

        # Get path points or centroid
        if 'path_points' in polygon_data:
            path_points = polygon_data['path_points']
            if len(path_points) < 3:
                continue  # A valid polygon needs at least 3 points

            # Extract x, y coordinates (ignore z if present)
            exterior_coords = [coord[:2] for coord in path_points]
            centroid = calculate_centroid(exterior_coords)
        elif 'centroid' in polygon_data:
            # Simplified format with direct centroid
            centroid = np.array(polygon_data['centroid'][:2])
        else:
            continue

        features_list.append({
            'filename': json_name,
            'category': category,
            'centroid': centroid.tolist(),
            'score': score
        })

    return features_list


def parse_nuclei_json(json_path: Path) -> List[Dict]:
    """
    Parse nuclei detection JSON file and extract features.

    Returns list of nuclei features with centroids, categories, and scores.
    """
    json_name = json_path.name

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    features = extract_features_from_json(json_data, json_name)
    return features


def calculate_centroid_distance(gt_features: List[Dict], pred_features: List[Dict], distance_threshold: float = 15.0) -> List[Dict]:
    """
    Match predictions to ground truth based on centroid distance (official PUMA method).

    This follows the exact matching strategy from the official evaluation code:
    1. Organize predictions by category
    2. For each GT nucleus, find eligible predictions within distance threshold
    3. Sort eligible predictions by confidence (descending) and distance (ascending)
    4. Take the best match
    5. Remove matched prediction from pool (one-to-one matching)

    Args:
        gt_features: List of ground truth nuclei features
        pred_features: List of predicted nuclei features
        distance_threshold: Maximum centroid distance for matching (default 15 pixels)

    Returns:
        List of matched predictions with their metrics
    """
    results = []
    pred_structure = {}

    # Organize pred_features into a dictionary by category for faster access
    for pred_feature in pred_features:
        match_key = pred_feature['category']
        if match_key not in pred_structure:
            pred_structure[match_key] = []
        pred_structure[match_key].append(pred_feature)

    # For each ground truth, find the best matching prediction
    for gt_feature in gt_features:
        match_key = gt_feature['category']
        eligible_predictions = []

        # Check if there are predictions matching the same category
        if match_key in pred_structure:
            for pred_feature in pred_structure[match_key]:
                # Calculate the Euclidean distance between ground truth and prediction centroids
                distance = np.linalg.norm(
                    np.array(gt_feature['centroid']) - np.array(pred_feature['centroid'])
                )

                # Filter predictions based on distance threshold
                # Use <= to handle exact matches (distance = 0.0) correctly
                if distance <= distance_threshold:
                    eligible_predictions.append({
                        'pred_json': pred_feature['filename'],
                        'gt_category': gt_feature['category'],
                        'pred_category': pred_feature['category'],
                        'distance': distance,
                        'pred_score': pred_feature['score'],
                        'pred_feature': pred_feature,
                    })

        # Sort eligible predictions by descending score and ascending distance
        eligible_predictions.sort(key=lambda x: (-x['pred_score'], x['distance']))

        # If we have any eligible prediction, take the best match
        if eligible_predictions:
            best_match = eligible_predictions[0]
            results.append(best_match)

            # Find and remove the used prediction from pred_structure
            for i, pred in enumerate(pred_structure[match_key]):
                if np.array_equal(pred['centroid'], best_match['pred_feature']['centroid']):
                    del pred_structure[match_key][i]
                    break

    return results


def calculate_classification_metrics(results: List[Dict], gt_features: List[Dict], pred_features: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate precision, recall, and F1 score per class based on matching results.

    Args:
        results: List of matched predictions
        gt_features: List of all ground truth features
        pred_features: List of all predicted features

    Returns:
        Dictionary with metrics per class and micro/macro aggregates
    """
    # Extract true positive categories (matched predictions)
    pred_tp = [match['pred_category'] for match in results]

    # Ground truth categories
    ground_truth = [feature['category'] for feature in gt_features]

    # All predicted categories
    pred_all = [feature['category'] for feature in pred_features]

    # Count occurrences of each category
    gt_dict = dict(zip(*np.unique(ground_truth, return_counts=True))) if ground_truth else {}
    pred_dict = dict(zip(*np.unique(pred_all, return_counts=True))) if pred_all else {}
    tp_dict = dict(zip(*np.unique(pred_tp, return_counts=True))) if pred_tp else {}

    micro_TP, micro_FP, micro_FN = 0, 0, 0
    results_metrics = {}

    # Calculate metrics for each category
    all_categories = set(list(gt_dict.keys()) + list(pred_dict.keys()))
    for category in all_categories:
        TP = tp_dict.get(category, 0)
        FP = pred_dict.get(category, 0) - TP
        FN = gt_dict.get(category, 0) - TP

        micro_TP += TP
        micro_FP += FP
        micro_FN += FN

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        # If class is absent in both GT and predictions (TP=0, FP=0, FN=0)
        # Score should be 0 (not informative) rather than 1.0 (perfect)
        if TP == 0 and FP == 0 and FN == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        results_metrics[category] = {
            'TP': int(TP), 'FP': int(FP), 'FN': int(FN),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score)
        }

    # Calculate micro metrics (aggregated across categories)
    micro_precision = micro_TP / (micro_TP + micro_FP) if micro_TP + micro_FP > 0 else 0
    micro_recall = micro_TP / (micro_TP + micro_FN) if micro_TP + micro_FN > 0 else 0
    micro_f1_score = 2 * micro_precision * micro_recall / (
        micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0

    # Calculate macro F1 (average of F1 scores per category)
    macro_f1_score = np.mean([metrics['f1_score'] for metrics in results_metrics.values()]) if results_metrics else 0

    results_metrics['micro'] = {
        'precision': float(micro_precision),
        'recall': float(micro_recall),
        'f1_score': float(micro_f1_score)
    }
    results_metrics['macro'] = {
        'f1_score': float(macro_f1_score)
    }

    return results_metrics


def evaluate_files(ground_truth_file: Path, pred_file: Path) -> Dict[str, Dict]:
    """
    Evaluate predictions against ground truth for a single file.

    Args:
        ground_truth_file: Path to ground truth JSON
        pred_file: Path to prediction JSON

    Returns:
        Dictionary with metrics per class and aggregates
    """
    gt_features = parse_nuclei_json(ground_truth_file)
    pred_features = parse_nuclei_json(pred_file)

    results = calculate_centroid_distance(gt_features, pred_features)
    metrics = calculate_classification_metrics(results, gt_features, pred_features)

    return metrics


def grade(submission: pd.DataFrame, answers: pd.DataFrame, submission_dir: Optional[Path] = None, answers_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Grade a PUMA Track 2 Task 2 submission (Nuclei Detection - 10 Classes).

    Evaluation follows the official PUMA Challenge methodology:
    - Match predictions to ground truth using 15-pixel centroid distance
    - Use confidence scores to prioritize matches
    - Compute F1 score per class (tumor, lymphocytes, plasma_cells, histiocytes, melanophages, neutrophils, stromal_cells, epithelium, endothelium, apoptotic_cells)
    - Compute macro F1 score (average across classes) - PRIMARY METRIC

    Args:
        submission: Submission DataFrame with columns: case_id, predicted_nuclei_path
        answers: Ground truth DataFrame with columns: case_id, image_path, label_path
        submission_dir: Directory containing submission files
        answers_dir: Directory containing ground truth files

    Returns:
        Dictionary with macro_f1 (primary metric) and per-class metrics
    """
    if submission_dir is None:
        raise InvalidSubmissionError("Detection task requires submission_dir")

    if answers_dir is None:
        raise InvalidSubmissionError("Detection task requires answers_dir")

    # Validate submission format
    if 'case_id' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'case_id' column")

    if 'predicted_nuclei_path' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'predicted_nuclei_path' column")

    # Check answers format
    assert 'case_id' in answers.columns
    assert 'label_path' in answers.columns

    # Track 2 has 10 classes (granular classification)
    expected_classes = {
        'tumor', 'lymphocytes', 'plasma_cells', 'histiocytes',
        'melanophages', 'neutrophils', 'stromal_cells',
        'epithelium', 'endothelium', 'apoptotic_cells'
    }

    # Aggregate metrics across all cases
    f1_scores_per_class = {}
    num_cases = 0

    for idx, row in submission.iterrows():
        case_id = row['case_id']
        pred_path = submission_dir / row['predicted_nuclei_path']

        # Find ground truth
        gt_row = answers[answers['case_id'] == case_id]
        if gt_row.empty:
            raise InvalidSubmissionError(f"No ground truth found for {case_id}")

        gt_path = answers_dir / gt_row.iloc[0]['label_path']

        # Load predictions
        if not pred_path.exists():
            raise InvalidSubmissionError(f"Predicted nuclei file not found: {pred_path}")

        try:
            # Evaluate this case using official PUMA method
            case_metrics = evaluate_files(gt_path, pred_path)
        except Exception as e:
            raise InvalidSubmissionError(f"Error evaluating case {case_id}: {e}")

        # Accumulate per-class F1 scores
        for class_name in expected_classes:
            if class_name in case_metrics:
                if class_name not in f1_scores_per_class:
                    f1_scores_per_class[class_name] = []
                f1_scores_per_class[class_name].append(case_metrics[class_name]['f1_score'])

        num_cases += 1

    if num_cases == 0:
        raise InvalidSubmissionError("No cases were evaluated")

    # Compute average F1 score per class across all cases
    avg_f1_per_class = {}
    for class_name in expected_classes:
        if class_name in f1_scores_per_class and f1_scores_per_class[class_name]:
            avg_f1_per_class[class_name] = np.mean(f1_scores_per_class[class_name])
        else:
            # If class never appears in any case, treat as zero to avoid inflated metrics
            avg_f1_per_class[class_name] = 0.0

    # Compute macro F1 (average across classes) - PRIMARY METRIC
    macro_f1 = np.mean(list(avg_f1_per_class.values()))

    # Print summary
    print(f"\n=== PUMA Track 2 Task 2 (Nuclei Detection - 10 Classes) Results ===")
    print(f"Cases evaluated: {num_cases}")
    print(f"Per-class F1 scores (averaged across cases):")
    for class_name in sorted(expected_classes):
        f1 = avg_f1_per_class.get(class_name, 0.0)
        print(f"  {class_name}: {f1:.4f}")
    print(f"Macro F1 Score (Primary): {macro_f1:.4f}")

    # Build results dictionary
    results_dict = {
        "f1": float(macro_f1),
    }

    # Add per-class F1 scores to result
    for class_name, f1 in avg_f1_per_class.items():
        results_dict[f"f1_{class_name}"] = float(f1)

    # Calculate position and mean position if leaderboard exists
    mean_position_value = None
    leaderboard_path = Path(__file__).parent / 'leaderboard.csv'

    if leaderboard_path.exists():
        try:
            leaderboard_df = pd.read_csv(leaderboard_path)
            submission_dict = {"f1": results_dict["f1"]}
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
        # Fallback: use negative f1 for ranking (higher f1 = lower/better rank)
        results_dict['overall'] = -results_dict['f1']

    return results_dict
