"""
Grading script for PUMA Track 1 Task 2 - Nuclei Detection (3 Classes)

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

Classes (3):
- tumor
- TILs (lymphocytes + plasma cells)
- other (histiocytes, melanophages, neutrophils, stromal, epithelium, endothelium, apoptotic)

Based on official PUMA evaluation code from:
https://github.com/PUMA-Challenge/PUMA-challenge-eval-track1
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from rexmle.grade_helpers import InvalidSubmissionError


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
        # Get category/class name
        category = polygon_data.get('name', polygon_data.get('class'))
        if category is None:
            continue
        
        # Normalize category name to match expected classes (case-insensitive matching)
        # Expected classes: 'tumor', 'TILs', 'other'
        category_lower = category.lower().strip()
        if category_lower == 'tils' or category_lower == 'til' or category_lower == 'lymphocyte' or category_lower == 'lymphocytes':
            category = 'TILs'
        elif category_lower == 'tumor' or category_lower == 'tumour':
            category = 'tumor'
        elif category_lower == 'other' or category_lower == 'others':
            category = 'other'
        # Otherwise keep original category name

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
                if distance < distance_threshold:
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


def calculate_classification_metrics(results: List[Dict], gt_features: List[Dict], pred_features: List[Dict], expected_classes: Optional[set] = None) -> Dict[str, Dict]:
    """
    Calculate precision, recall, and F1 score per class based on matching results.

    Args:
        results: List of matched predictions
        gt_features: List of all ground truth features
        pred_features: List of all predicted features
        expected_classes: Set of expected class names (to ensure all classes are included in metrics)

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
    # Include expected classes even if they don't appear in this case
    all_categories = set(list(gt_dict.keys()) + list(pred_dict.keys()))
    if expected_classes:
        all_categories = all_categories.union(expected_classes)
    for category in all_categories:
        TP = tp_dict.get(category, 0)
        FP = pred_dict.get(category, 0) - TP
        FN = gt_dict.get(category, 0) - TP

        micro_TP += TP
        micro_FP += FP
        micro_FN += FN

        # Handle edge case: when class is absent in both GT and prediction
        # Score should be 0 (not informative) rather than 1.0 (perfect)
        if TP == 0 and FP == 0 and FN == 0:
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
        else:
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
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


def evaluate_files(ground_truth_file: Path, pred_file: Path, expected_classes: Optional[set] = None) -> Dict[str, Dict]:
    """
    Evaluate predictions against ground truth for a single file.

    Args:
        ground_truth_file: Path to ground truth JSON
        pred_file: Path to prediction JSON
        expected_classes: Set of expected class names (to ensure all classes are included)

    Returns:
        Dictionary with metrics per class and aggregates
    """
    gt_features = parse_nuclei_json(ground_truth_file)
    pred_features = parse_nuclei_json(pred_file)

    results = calculate_centroid_distance(gt_features, pred_features)
    metrics = calculate_classification_metrics(results, gt_features, pred_features, expected_classes)

    return metrics


def grade(submission: pd.DataFrame, answers: pd.DataFrame, submission_dir: Optional[Path] = None, answers_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Grade a PUMA Track 1 Task 2 submission (Nuclei Detection - 3 Classes).

    Evaluation follows the official PUMA Challenge methodology:
    - Match predictions to ground truth using 15-pixel centroid distance
    - Use confidence scores to prioritize matches
    - Compute F1 score per class (tumor, TILs, other)
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

    # Track 1 has 3 classes
    expected_classes = {'tumor', 'TILs', 'other'}

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
            # Pass expected_classes to ensure all classes are included in metrics
            case_metrics = evaluate_files(gt_path, pred_path, expected_classes)
        except Exception as e:
            raise InvalidSubmissionError(f"Error evaluating case {case_id}: {e}")

        # Accumulate per-class F1 scores (all expected classes should now be in case_metrics)
        for class_name in expected_classes:
            if class_name in case_metrics:
                if class_name not in f1_scores_per_class:
                    f1_scores_per_class[class_name] = []
                f1_scores_per_class[class_name].append(case_metrics[class_name]['f1_score'])
            else:
                # If class is missing from case_metrics, it means it wasn't in expected_classes
                # but we should still track it (this shouldn't happen with the fix above)
                if class_name not in f1_scores_per_class:
                    f1_scores_per_class[class_name] = []
                f1_scores_per_class[class_name].append(1.0)  # Perfect match when absent

        num_cases += 1

    if num_cases == 0:
        raise InvalidSubmissionError("No cases were evaluated")

    # Compute average F1 score per class across all cases
    avg_f1_per_class = {}
    for class_name in expected_classes:
        if class_name in f1_scores_per_class and f1_scores_per_class[class_name]:
            avg_f1_per_class[class_name] = np.mean(f1_scores_per_class[class_name])
        else:
            avg_f1_per_class[class_name] = 0.0

    # Compute macro F1 (average across classes) - PRIMARY METRIC
    macro_f1 = np.mean(list(avg_f1_per_class.values()))

    # Print summary
    print(f"\n=== PUMA Track 1 Task 2 (Nuclei Detection - 3 Classes) Results ===")
    print(f"Cases evaluated: {num_cases}")
    print(f"Per-class F1 scores (averaged across cases):")
    for class_name in sorted(expected_classes):
        f1 = avg_f1_per_class.get(class_name, 0.0)
        print(f"  {class_name}: {f1:.4f}")
    print(f"Macro F1 Score (Primary): {macro_f1:.4f}")

    result = {
        "macro_f1": float(macro_f1),
    }

    # Add per-class F1 scores to result
    for class_name, f1 in avg_f1_per_class.items():
        result[f"f1_{class_name}"] = float(f1)

    # Calculate position and mean position if leaderboard exists
    mean_position_value = None
    leaderboard_path = Path(__file__).parent / 'leaderboard.csv'

    if leaderboard_path.exists():
        try:
            leaderboard_df = pd.read_csv(leaderboard_path)
            submission_dict = {"f1": result["macro_f1"]}
            positions = calculate_positions_and_mean(submission_dict, leaderboard_df)
            mean_position_value = positions.get('mean_position')

            # Add individual metric positions to results
            for metric_name, position_value in positions.items():
                if metric_name != 'mean_position':
                    result[metric_name] = position_value

            result['mean_position'] = mean_position_value

            # Calculate percentile: 1 - ((mean_position - 1) / num_competitors)
            num_competitors = len(leaderboard_df)
            percentile = 1 - ((mean_position_value - 1) / (num_competitors))
            result['percentile'] = percentile

            print(f"Mean Position: {mean_position_value:.2f}")
            print(f"Percentile: {percentile:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate position: {e}")

    # Set 'overall' for ranking (lower mean_position is better)
    if mean_position_value is not None:
        result['overall'] = mean_position_value
    else:
        # Fallback: use negative macro_f1 for ranking (higher f1 = lower/better rank)
        result['overall'] = -result['macro_f1']

    return result



# Metric direction configuration and position calculation
METRIC_DIRECTIONS = {
    "f1": "higher_better"
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
