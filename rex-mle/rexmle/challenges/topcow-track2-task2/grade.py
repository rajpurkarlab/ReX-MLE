"""
Grading script for TopCoW Track 2 Task 2 - MRA 3D Bounding Box Detection

Implements metrics:
- boundary_iou: Boundary Intersection over Union
- iou: Standard Intersection over Union

Based on TopCoW_Eval_Metrics reference implementation.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import math
from typing import Optional, Tuple

from rexmle.grade_helpers import InvalidSubmissionError

# Load metric directions
CHALLENGE_DIR = Path(__file__).parent
with open(CHALLENGE_DIR / "metric_config.json") as f:
    METRIC_DIRECTIONS = json.load(f)["metric_directions"]

# Constants from TopCoW_Eval_Metrics
BOUNDARY_DISTANCE_RATIO = 0.2
MAX_DISTANCE_RATIO = 0.5


def parse_roi_txt(roi_txt_path: Path) -> Tuple[list, list]:
    """Parse ROI metadata txt file."""
    with open(roi_txt_path) as f:
        lines = f.readlines()

    size_arr = re.findall(r"\b\d+\b", lines[1])
    location_arr = re.findall(r"\b\d+\b", lines[2])

    return ([int(x) for x in size_arr], [int(x) for x in location_arr])


def parse_roi_json(roi_json_path: Path) -> Tuple[list, list]:
    """Parse ROI size and location from json file."""
    with open(roi_json_path, mode="r", encoding="utf-8") as file:
        data = json.load(file)

    size_arr = data["size"]
    location_arr = data["location"]

    return ([int(x) for x in size_arr], [int(x) for x in location_arr])


def boundary_points_with_distances(size_arr: list, location_arr: list, distance_arr: list) -> set:
    """Generate boundary points with distances from bbox."""
    x_size, y_size, z_size = size_arr
    x_loc, y_loc, z_loc = location_arr
    x_dist, y_dist, z_dist = distance_arr

    # Calculate boundaries with distance
    x_min = x_loc - x_dist
    x_max = x_loc + x_size - 1 + x_dist
    y_min = y_loc - y_dist
    y_max = y_loc + y_size - 1 + y_dist
    z_min = z_loc - z_dist
    z_max = z_loc + z_size - 1 + z_dist

    boundary_points = set()

    # Generate all points within the boundary volume
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            for z in range(z_min, z_max + 1):
                boundary_points.add((x, y, z))

    return boundary_points


def iou_of_sets(first_set: set, second_set: set) -> float:
    """Compute intersection over union for two sets."""
    intersection = first_set.intersection(second_set)
    len_intersection = len(intersection)

    union = first_set.union(second_set)
    len_union = len(union)

    iou = len_intersection / len_union if len_union > 0 else 0
    return float(iou)


def boundary_iou_from_tuple(
    first_box: Tuple[list, list],
    second_box: Tuple[list, list],
    boundary_distance_ratio: float,
) -> float:
    """Calculate boundary IoU for two bounding boxes."""
    # Use a fixed ratio for boundary distances
    size_arr_1, loc_arr_1 = first_box
    dist_arr_1 = [math.ceil(s * boundary_distance_ratio) for s in size_arr_1]

    size_arr_2, loc_arr_2 = second_box
    dist_arr_2 = [math.ceil(s * boundary_distance_ratio) for s in size_arr_2]

    # Get boundary points for both boxes
    first_boundary = boundary_points_with_distances(
        size_arr=size_arr_1,
        location_arr=loc_arr_1,
        distance_arr=dist_arr_1,
    )
    second_boundary = boundary_points_with_distances(
        size_arr=size_arr_2,
        location_arr=loc_arr_2,
        distance_arr=dist_arr_2,
    )

    # Get the IoU of the boundary sets
    boundary_iou = iou_of_sets(first_boundary, second_boundary)

    return boundary_iou


def calculate_positions_and_mean(submission_metrics: dict, leaderboard_df: pd.DataFrame) -> dict:
    """
    Calculate position for each metric and mean position across all metrics.

    Args:
        submission_metrics: dict with metric values
        leaderboard_df: DataFrame with existing leaderboard data

    Returns:
        dict with metric_name: position pairs and 'mean_position'
    """
    # Create temp dataframe with leaderboard + new submission
    temp_df = pd.concat([leaderboard_df, pd.DataFrame([submission_metrics])], ignore_index=True)

    # Get metric columns (exclude rank, team, mean_position, and _position columns)
    metric_columns = [col for col in temp_df.columns
                     if col not in ['rank', 'team', 'mean_position']
                     and not col.endswith('_position')]

    positions = {}

    for metric in metric_columns:
        if metric in METRIC_DIRECTIONS:
            direction = METRIC_DIRECTIONS[metric]

            if direction == 'lower_better':
                temp_df[f'{metric}_position'] = temp_df[metric].rank(ascending=True, method='min')
            else:  # higher_better
                temp_df[f'{metric}_position'] = temp_df[metric].rank(ascending=False, method='min')

            positions[f'{metric}_position'] = temp_df[f'{metric}_position'].iloc[-1]

    # Calculate mean position
    if positions:
        positions['mean_position'] = sum(positions.values()) / len(positions)

    return positions


def iou_dict_from_files(first_box_path: Path, second_box_path: Path) -> dict:
    """Compute IoU and Boundary IoU from two bbox files."""
    assert first_box_path.is_file(), f"first_box_path doesn't exist: {first_box_path}"
    assert second_box_path.is_file(), f"second_box_path doesn't exist: {second_box_path}"

    # Parse first box
    if first_box_path.suffix.lower() == ".txt":
        first_box = parse_roi_txt(first_box_path)
    elif first_box_path.suffix.lower() == ".json":
        first_box = parse_roi_json(first_box_path)
    else:
        raise ValueError(f"{first_box_path} invalid roi file extension!")

    # Parse second box
    if second_box_path.suffix.lower() == ".txt":
        second_box = parse_roi_txt(second_box_path)
    elif second_box_path.suffix.lower() == ".json":
        second_box = parse_roi_json(second_box_path)
    else:
        raise ValueError(f"{second_box_path} invalid roi file extension!")

    iou_dict = {}

    # Compute boundary IoU with specified ratio
    boundary_iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        BOUNDARY_DISTANCE_RATIO,
    )

    # Compute standard IoU (boundary ratio = 0.5)
    iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        MAX_DISTANCE_RATIO,
    )

    iou_dict["Boundary_IoU"] = boundary_iou
    iou_dict["IoU"] = iou

    return iou_dict


def grade(submission: pd.DataFrame, answers: pd.DataFrame, submission_dir: Optional[Path] = None, answers_dir: Optional[Path] = None) -> dict:
    """
    Grade TopCoW Track 2 Task 2 submission using boundary IoU and standard IoU.

    Returns all metrics from leaderboard:
    - boundary_iou, iou
    """
    if submission_dir is None:
        raise InvalidSubmissionError("Object detection task requires submission_dir")

    if answers_dir is None:
        raise InvalidSubmissionError("Object detection task requires answers_dir")

    # Validate submission format
    if 'image_id' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'image_id' column")
    if 'predicted_bbox_path' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'predicted_bbox_path' column")

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
    all_boundary_iou = []
    all_iou = []

    for _, row in submission.iterrows():
        image_id = str(row['image_id'])
        pred_box_path = submission_dir / row['predicted_bbox_path']

        # Find ground truth
        gt_row = answers[answers['image_id'].astype(str) == image_id]
        if gt_row.empty:
            raise InvalidSubmissionError(f"No ground truth found for {image_id}")

        gt_box_path = answers_dir / gt_row.iloc[0]['label_path']

        # Validate files exist
        if not pred_box_path.exists():
            raise InvalidSubmissionError(f"Predicted box not found: {pred_box_path}")
        if not gt_box_path.exists():
            raise InvalidSubmissionError(f"Ground truth box not found: {gt_box_path}")

        try:
            # Compute IoU metrics
            iou_dict = iou_dict_from_files(gt_box_path, pred_box_path)

            all_boundary_iou.append(iou_dict["Boundary_IoU"])
            all_iou.append(iou_dict["IoU"])

        except Exception as e:
            raise InvalidSubmissionError(f"Error evaluating {image_id}: {e}")

    # Aggregate metrics
    if len(all_iou) == 0:
        raise InvalidSubmissionError("No metrics computed")

    mean_boundary_iou = float(np.mean(all_boundary_iou))
    mean_iou = float(np.mean(all_iou))

    print(f"\n=== TopCoW Track 2 Task 2 Results ===")
    print(f"Cases evaluated: {len(all_iou)}")
    print(f"Boundary IoU: {mean_boundary_iou:.4f}")
    print(f"IoU: {mean_iou:.4f}")

    # Create results dict
    results_dict = {
        "boundary_iou": mean_boundary_iou,
        "iou": mean_iou,
    }

    # Calculate positions and mean position
    leaderboard_path = CHALLENGE_DIR / "leaderboard.csv"
    if leaderboard_path.exists():
        try:
            leaderboard_df = pd.read_csv(leaderboard_path)
            positions = calculate_positions_and_mean(results_dict, leaderboard_df)
            mean_position_value = positions.get('mean_position')

            if mean_position_value is not None:
                # Add individual metric positions to results
                for metric_name, position_value in positions.items():
                    if metric_name != 'mean_position':
                        results_dict[metric_name] = position_value

                results_dict['mean_position'] = mean_position_value

                # Calculate percentile: 1 - ((mean_position - 1) / num_competitors)
                num_competitors = len(leaderboard_df)
                percentile = 1 - ((mean_position_value - 1) / (num_competitors))
                results_dict['percentile'] = percentile

                results_dict['overall'] = mean_position_value
                print(f"Mean position: {mean_position_value:.2f}")
                print(f"Percentile: {percentile:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate positions: {e}")
            # Fall back to negative mean of metrics as overall score
            results_dict['overall'] = -(mean_boundary_iou + mean_iou) / 2
    else:
        # Fall back to negative mean of metrics as overall score
        results_dict['overall'] = -(mean_boundary_iou + mean_iou) / 2

    return results_dict
