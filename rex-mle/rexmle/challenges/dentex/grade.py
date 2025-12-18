"""
Grading script for DENTEX Challenge

This script evaluates object detection submissions using COCO evaluation metrics.

The challenge evaluates three aspects:
1. Quadrant detection (Q: 1-4)
2. Tooth enumeration (N: 1-8 within each quadrant)
3. Diagnosis classification (D: caries, deep caries, periapical lesions, impacted)

Metrics computed for each aspect:
- AP (Average Precision)
- AP50 (AP at IoU=0.50)
- AP75 (AP at IoU=0.75)
- AR (Average Recall)

Final score is the mean of AP across all three aspects.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json
import tempfile

from rexmle.grade_helpers import InvalidSubmissionError

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. Install with: pip install pycocotools")


def convert_bbox_format(bbox: List[float]) -> List[float]:
    """
    Convert bounding box from [x1, y1, x2, y2] to COCO format [x, y, width, height].

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format

    Returns:
        Bounding box in [x, y, width, height] format
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def separate_annotations_by_category(
    gt_data: Dict,
    pred_data: List[Dict],
    category_type: str,
    category_mapping: Dict[str, int]
) -> tuple:
    """
    Separate ground truth and predictions by category type (quadrant, enumeration, diagnosis).

    Args:
        gt_data: Ground truth COCO-format annotations
        pred_data: Prediction annotations
        category_type: One of 'quadrant', 'enumeration', 'diagnosis'
        category_mapping: Mapping from category names to IDs

    Returns:
        Tuple of (gt_coco_dict, pred_list) separated by category type
    """
    # Create separate ground truth for this category type
    gt_separated = {
        'images': gt_data['images'],
        'annotations': [],
        'categories': [],
        'info': gt_data.get('info', {'description': 'DENTEX Challenge'}),
        'licenses': gt_data.get('licenses', [])
    }

    # Map category type to the appropriate field in annotations
    category_field_map = {
        'quadrant': 'category_id_1',  # or 'quadrant_id'
        'enumeration': 'category_id_2',  # or 'enumeration_id'
        'diagnosis': 'category_id_3'  # or 'diagnosis_id'
    }

    # Process ground truth annotations
    for anno in gt_data['annotations']:
        # Extract the category ID for this specific aspect
        # Try both naming conventions: quadrant_id/category_id_1, etc.
        if category_type == 'quadrant':
            # Quadrant is the first digit (1-4)
            cat_id = anno.get('category_id_1')
            if cat_id is None:
                cat_id = anno.get('quadrant_id', 1)
        elif category_type == 'enumeration':
            # Enumeration is the second digit (1-8)
            cat_id = anno.get('category_id_2')
            if cat_id is None:
                cat_id = anno.get('enumeration_id', 1)
        else:  # diagnosis
            # Diagnosis is the third digit (1-4)
            cat_id = anno.get('category_id_3')
            if cat_id is None:
                cat_id = anno.get('diagnosis_id', 1)

        anno_copy = anno.copy()
        anno_copy['category_id'] = cat_id
        gt_separated['annotations'].append(anno_copy)

    # Create categories for this type
    gt_separated['categories'] = [
        {'id': cat_id, 'name': cat_name}
        for cat_name, cat_id in category_mapping.items()
    ]

    # Process predictions
    pred_separated = []
    for pred in pred_data:
        # Extract category ID for this aspect from prediction
        # Try both naming conventions: quadrant_id/category_id_1, etc.
        if category_type == 'quadrant':
            cat_id = pred.get('category_id_1')
            if cat_id is None:
                cat_id = pred.get('quadrant_id', 1)
        elif category_type == 'enumeration':
            cat_id = pred.get('category_id_2')
            if cat_id is None:
                cat_id = pred.get('enumeration_id', 1)
        else:  # diagnosis
            cat_id = pred.get('category_id_3')
            if cat_id is None:
                cat_id = pred.get('diagnosis_id', 1)

        pred_separated.append({
            'image_id': pred['image_id'],
            'category_id': cat_id,
            'bbox': pred['bbox'],
            'score': pred.get('score', pred.get('confidence', 0.5))
        })

    return gt_separated, pred_separated


def evaluate_category(
    gt_coco_dict: Dict,
    pred_list: List[Dict]
) -> Dict[str, float]:
    """
    Evaluate predictions for a single category type using COCO metrics.

    Args:
        gt_coco_dict: Ground truth in COCO format
        pred_list: List of predictions

    Returns:
        Dictionary with AP, AP50, AP75, AR metrics
    """
    if not PYCOCOTOOLS_AVAILABLE:
        raise ImportError("pycocotools is required for DENTEX evaluation")

    # Create temporary files for COCO evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
        json.dump(gt_coco_dict, gt_file)
        gt_path = gt_file.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
        json.dump(pred_list, pred_file)
        pred_path = pred_file.name

    try:
        # Load ground truth
        coco_gt = COCO(gt_path)

        # Load predictions
        coco_dt = coco_gt.loadRes(pred_path)

        # Run COCO evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            'AP': float(coco_eval.stats[0]),      # AP @ IoU=0.50:0.95
            'AP50': float(coco_eval.stats[1]),    # AP @ IoU=0.50
            'AP75': float(coco_eval.stats[2]),    # AP @ IoU=0.75
            'AR': float(coco_eval.stats[8])       # AR @ IoU=0.50:0.95
        }

    finally:
        # Clean up temporary files
        Path(gt_path).unlink(missing_ok=True)
        Path(pred_path).unlink(missing_ok=True)

    return metrics


def grade(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    submission_dir: Optional[Path] = None,
    answers_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Grade DENTEX submission using COCO evaluation metrics.

    The submission CSV should have columns:
    - image_id: Image identifier
    - predictions_json: Path to JSON file with predictions

    The JSON files should contain COCO-format predictions with:
    - image_id: Image ID
    - bbox: [x, y, width, height]
    - quadrant_id: Quadrant (1-4)
    - enumeration_id: Tooth number (1-8)
    - diagnosis_id: Diagnosis class (1-4)
    - score/confidence: Detection confidence

    Args:
        submission: Submission DataFrame
        answers: Ground truth DataFrame with image_id and ground_truth_json
        submission_dir: Directory containing submission files
        answers_dir: Directory containing ground truth files

    Returns:
        Dictionary with all metrics matching leaderboard columns:
        - ap_mean: Mean AP across all three aspects (primary metric)
        - ap_quadrant, ap50_quadrant, ap75_quadrant, ar_quadrant: Quadrant metrics
        - ap_enumeration, ap50_enumeration, ap75_enumeration, ar_enumeration: Enumeration metrics
        - ap_diagnosis, ap50_diagnosis, ap75_diagnosis, ar_diagnosis: Diagnosis metrics
    """
    if not PYCOCOTOOLS_AVAILABLE:
        raise InvalidSubmissionError(
            "pycocotools is required for DENTEX evaluation. "
            "Install with: pip install pycocotools"
        )

    if submission_dir is None:
        raise InvalidSubmissionError("submission_dir is required for DENTEX evaluation")

    if answers_dir is None:
        raise InvalidSubmissionError("answers_dir is required for DENTEX evaluation")

    # Validate submission format
    if 'image_id' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'image_id' column")

    if 'predictions_json' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'predictions_json' column")

    # Check that all images are present
    submission_ids = set(submission['image_id'])
    answer_ids = set(answers['image_id'])

    if submission_ids != answer_ids:
        missing = answer_ids - submission_ids
        if missing:
            raise InvalidSubmissionError(
                f"Missing predictions for images: {list(missing)[:5]}..."
            )

    # Load all ground truth JSON files (one per image)
    # Combine them into a single COCO-format dictionary
    gt_data = {
        'images': [],
        'annotations': [],
        'categories_1': None,
        'categories_2': None,
        'categories_3': None
    }

    for idx, row in answers.iterrows():
        image_id = row['image_id']
        gt_json_path = answers_dir / row['ground_truth_json']

        if not gt_json_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {gt_json_path}"
            )

        # Load individual ground truth JSON
        with open(gt_json_path, 'r') as f:
            image_gt = json.load(f)

        # Merge into combined gt_data
        gt_data['images'].extend(image_gt['images'])
        gt_data['annotations'].extend(image_gt['annotations'])

        # Store categories from first image (should be same for all)
        if gt_data['categories_1'] is None and 'categories_1' in image_gt:
            gt_data['categories_1'] = image_gt['categories_1']
        if gt_data['categories_2'] is None and 'categories_2' in image_gt:
            gt_data['categories_2'] = image_gt['categories_2']
        if gt_data['categories_3'] is None and 'categories_3' in image_gt:
            gt_data['categories_3'] = image_gt['categories_3']

    # Create mapping from image filenames to numeric IDs
    filename_to_id = {img['file_name'].replace('.png', ''): img['id'] for img in gt_data['images']}

    # Collect all predictions
    all_predictions = []

    for idx, row in submission.iterrows():
        image_id = row['image_id']
        pred_json_path = submission_dir / row['predictions_json']

        if not pred_json_path.exists():
            raise InvalidSubmissionError(
                f"Prediction file not found: {pred_json_path}"
            )

        # Load predictions for this image
        with open(pred_json_path, 'r') as f:
            image_predictions = json.load(f)

        # Ensure predictions are in list format
        if isinstance(image_predictions, dict):
            if 'annotations' in image_predictions:
                image_predictions = image_predictions['annotations']
            else:
                image_predictions = [image_predictions]

        # Convert image_id from string to numeric if needed
        numeric_image_id = filename_to_id.get(image_id, image_id)
        if isinstance(numeric_image_id, str):
            # Try to extract numeric ID if string
            try:
                numeric_image_id = int(image_id.split('_')[-1])
            except:
                numeric_image_id = image_id

        # Add/update image_id and convert to numeric format
        for pred in image_predictions:
            # Convert string image_id to numeric
            if 'image_id' in pred and isinstance(pred['image_id'], str):
                pred_img_id = pred['image_id'].replace('.png', '')
                pred['image_id'] = filename_to_id.get(pred_img_id, numeric_image_id)
            elif 'image_id' not in pred:
                pred['image_id'] = numeric_image_id

            # Ensure bbox is in COCO format [x, y, width, height]
            if 'bbox' in pred and len(pred['bbox']) == 4:
                bbox = pred['bbox']
                # Check if it's in [x1, y1, x2, y2] format (x2 > x1 + width)
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    # Might be in corner format, convert if width/height seem large
                    if bbox[2] > bbox[0] + 50 or bbox[3] > bbox[1] + 50:
                        pred['bbox'] = convert_bbox_format(bbox)

        all_predictions.extend(image_predictions)

    # Category mappings
    category_mappings = {
        'quadrant': {f'Q{i}': i for i in range(1, 5)},  # Q1, Q2, Q3, Q4
        'enumeration': {f'N{i}': i for i in range(1, 9)},  # N1-N8
        'diagnosis': {
            'caries': 1,
            'deep_caries': 2,
            'periapical_lesion': 3,
            'impacted': 4
        }
    }

    # Evaluate each category separately
    results = {}

    for category_type in ['quadrant', 'enumeration', 'diagnosis']:
        gt_separated, pred_separated = separate_annotations_by_category(
            gt_data,
            all_predictions,
            category_type,
            category_mappings[category_type]
        )

        metrics = evaluate_category(gt_separated, pred_separated)
        results[category_type] = metrics

    # Compute aggregate scores (mean across all categories)
    mean_ap = np.mean([
        results['quadrant']['AP'],
        results['enumeration']['AP'],
        results['diagnosis']['AP']
    ])

    mean_ap50 = np.mean([
        results['quadrant']['AP50'],
        results['enumeration']['AP50'],
        results['diagnosis']['AP50']
    ])

    mean_ap75 = np.mean([
        results['quadrant']['AP75'],
        results['enumeration']['AP75'],
        results['diagnosis']['AP75']
    ])

    mean_ar = np.mean([
        results['quadrant']['AR'],
        results['enumeration']['AR'],
        results['diagnosis']['AR']
    ])

    # Print detailed results
    print("\n=== DENTEX Evaluation Results ===")
    for category_type in ['quadrant', 'enumeration', 'diagnosis']:
        print(f"\n{category_type.capitalize()}:")
        for metric, value in results[category_type].items():
            print(f"  {metric}: {value:.4f}")

    print("\n=== Mean Metrics Across Categories ===")
    print(f"  AP (Mean): {mean_ap:.4f}")
    print(f"  AP50 (Mean): {mean_ap50:.4f}")
    print(f"  AP75 (Mean): {mean_ap75:.4f}")
    print(f"  AR (Mean): {mean_ar:.4f}")

    print(f"\n=== Mean AP (Final Score): {mean_ap:.4f} ===\n")

    # Build results dictionary
    results_dict = {
        'ap_mean': mean_ap,
        'ap50_mean': mean_ap50,
        'ap75_mean': mean_ap75,
        'ar_mean': mean_ar,
        'ap_quadrant': results['quadrant']['AP'],
        'ap50_quadrant': results['quadrant']['AP50'],
        'ap75_quadrant': results['quadrant']['AP75'],
        'ar_quadrant': results['quadrant']['AR'],
        'ap_enumeration': results['enumeration']['AP'],
        'ap50_enumeration': results['enumeration']['AP50'],
        'ap75_enumeration': results['enumeration']['AP75'],
        'ar_enumeration': results['enumeration']['AR'],
        'ap_diagnosis': results['diagnosis']['AP'],
        'ap50_diagnosis': results['diagnosis']['AP50'],
        'ap75_diagnosis': results['diagnosis']['AP75'],
        'ar_diagnosis': results['diagnosis']['AR'],
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
    # If mean_position not available, fall back to negated mean_ap (higher AP = better = lower rank)
    if mean_position_value is not None:
        results_dict['overall'] = mean_position_value
    else:
        results_dict['overall'] = -mean_ap  # Negative so higher AP gives lower (better) score

    return results_dict



# Metric direction configuration and position calculation
METRIC_DIRECTIONS = {
    "ap_mean": "higher_better",
    "ap50_mean": "higher_better",
    "ap75_mean": "higher_better",
    "ar_mean": "higher_better",
    "ap_quadrant": "higher_better",
    "ap50_quadrant": "higher_better",
    "ap75_quadrant": "higher_better",
    "ar_quadrant": "higher_better",
    "ap_enumeration": "higher_better",
    "ap50_enumeration": "higher_better",
    "ap75_enumeration": "higher_better",
    "ar_enumeration": "higher_better",
    "ap_diagnosis": "higher_better",
    "ap50_diagnosis": "higher_better",
    "ap75_diagnosis": "higher_better",
    "ar_diagnosis": "higher_better"
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
