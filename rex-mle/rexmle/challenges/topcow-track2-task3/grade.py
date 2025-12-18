"""
Grading script for TopCoW Track 2 Task 3 - MRA Graph Edge Classification

Task 3 requires edge classifications; predictions may be YAML (.yml/.yaml) or JSON (.json) files with anterior/posterior edge classifications.
Ground truth are .yml files (edge lists).

Implements metrics:
- anterior_accuracy: Variant-balanced anterior topology accuracy
- posterior_accuracy: Variant-balanced posterior topology accuracy

Based on TopCoW_Eval_Metrics reference implementation.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
import SimpleITK as sitk
import yaml
import json
from skimage import measure
from sklearn.metrics import balanced_accuracy_score

from rexmle.grade_helpers import InvalidSubmissionError

# Load metric directions
CHALLENGE_DIR = Path(__file__).parent
with open(CHALLENGE_DIR / "metric_config.json") as f:
    METRIC_DIRECTIONS = json.load(f)["metric_directions"]

# Constants from TopCoW_Eval_Metrics
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


def extract_labels(array: np.ndarray) -> list:
    """Extract unique labels WITHOUT background 0."""
    labels = np.unique(array)
    filtered_labels = [int(x) for x in labels[labels != 0]]
    return filtered_labels


def get_label_by_name(label_name: str, label_map: dict) -> int:
    """Get label intensity int value by label-name."""
    return int([k for k, v in label_map.items() if v == label_name][0])


def filter_mask_by_label(mask: np.ndarray, label: int) -> np.ndarray:
    """Filter mask keeping only voxels matching the label."""
    return np.where(mask == label, 1, 0)


def connected_components(img: np.ndarray) -> tuple:
    """Identify connected components and calculate B0 Betti number."""
    assert img.ndim == 3, "betti_number expects a 3D input"
    N26 = 3  # full connectivity

    b0_labels, b0 = measure.label(img, return_num=True, connectivity=N26)
    props = measure.regionprops(b0_labels)
    sizes = [obj.area for obj in props]
    sizes.sort()

    return int(b0), props, sizes


def get_coords_N26_ind(coords: np.ndarray) -> np.ndarray:
    """Get 26-neighborhood indices for a list of coords."""
    motion = np.transpose(np.indices((3, 3, 3)) - 1).reshape(-1, 3)
    motion = np.delete(motion, 13, axis=0)  # Remove origin

    all_N26_ind = np.expand_dims(coords[0], axis=0)

    for p in coords:
        N26_ind = motion + np.expand_dims(p, axis=0)
        all_N26_ind = np.concatenate((N26_ind, all_N26_ind))

    # De-duplicate
    unique_ind = np.unique(all_N26_ind, axis=0)

    # Remove indices already in coords
    neighbors_only = unique_ind[np.all(np.any((unique_ind - coords[:, None]), axis=2), axis=0)]

    return neighbors_only


def get_coords_N26_val(coords: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Get values of all neighbors of coords."""
    indices = get_coords_N26_ind(coords)
    return img[indices[:, 0], indices[:, 1], indices[:, 2]]


def get_label_neighbors(label_props, padded: np.ndarray) -> list:
    """Get neighboring labels for a label's regionprops."""
    neighbors = []

    for region in label_props:
        vals = get_coords_N26_val(region.coords, padded)
        unique_vals = np.unique(vals)
        neighbors.extend(unique_vals)

    neighbors = extract_labels(np.array(neighbors))
    return neighbors


def has_A1(mask_arr: np.ndarray, side: str = "L") -> bool:
    """Check if mask has A1 edge (ACA-ICA touch)."""
    padded = np.pad(mask_arr, pad_width=1)

    assert side in ["L", "R"], "unknown side"
    ACA = get_label_by_name(f"{side}-ACA", MUL_CLASS_LABEL_MAP)
    ICA = get_label_by_name(f"{side}-ICA", MUL_CLASS_LABEL_MAP)

    ACA_mask = filter_mask_by_label(padded, ACA)

    _, ACA_props, _ = connected_components(ACA_mask)
    ACA_neighbors = get_label_neighbors(ACA_props, padded)

    return ICA in ACA_neighbors


def has_P1(mask_arr: np.ndarray, side: str = "L") -> bool:
    """Check if mask has P1 edge (PCA-BA touch)."""
    padded = np.pad(mask_arr, pad_width=1)

    assert side in ["L", "R"], "unknown side"
    PCA = get_label_by_name(f"{side}-PCA", MUL_CLASS_LABEL_MAP)
    BA = get_label_by_name("BA", MUL_CLASS_LABEL_MAP)

    PCA_mask = filter_mask_by_label(padded, PCA)

    _, PCA_props, _ = connected_components(PCA_mask)
    PCA_neighbors = get_label_neighbors(PCA_props, padded)

    return BA in PCA_neighbors


def generate_edgelist_from_mask(mask_arr: np.ndarray) -> list:
    """
    Generate edge list of two (1,4) vectors from segmentation mask.

    Returns [anterior, posterior] where:
    - anterior = [L-A1, Acom, 3rd-A2, R-A1]
    - posterior = [L-Pcom, L-P1, R-P1, R-Pcom]
    """
    unique_labels = extract_labels(mask_arr)

    # Get label integers
    label_Acom = get_label_by_name("Acom", MUL_CLASS_LABEL_MAP)
    label_trd_A2 = get_label_by_name("3rd-A2", MUL_CLASS_LABEL_MAP)
    label_L_Pcom = get_label_by_name("L-Pcom", MUL_CLASS_LABEL_MAP)
    label_R_Pcom = get_label_by_name("R-Pcom", MUL_CLASS_LABEL_MAP)

    label_L_ACA = get_label_by_name("L-ACA", MUL_CLASS_LABEL_MAP)
    label_R_ACA = get_label_by_name("R-ACA", MUL_CLASS_LABEL_MAP)

    label_L_PCA = get_label_by_name("L-PCA", MUL_CLASS_LABEL_MAP)
    label_R_PCA = get_label_by_name("R-PCA", MUL_CLASS_LABEL_MAP)

    # Anterior edges
    L_A1 = label_L_ACA in unique_labels and has_A1(mask_arr, "L")
    Acom = label_Acom in unique_labels
    trd_A2 = label_trd_A2 in unique_labels
    R_A1 = label_R_ACA in unique_labels and has_A1(mask_arr, "R")

    # Posterior edges
    L_Pcom = label_L_Pcom in unique_labels
    L_P1 = label_L_PCA in unique_labels and has_P1(mask_arr, "L")
    R_P1 = label_R_PCA in unique_labels and has_P1(mask_arr, "R")
    R_Pcom = label_R_Pcom in unique_labels

    ant_list = [int(edge) for edge in (L_A1, Acom, trd_A2, R_A1)]
    pos_list = [int(edge) for edge in (L_Pcom, L_P1, R_P1, R_Pcom)]

    return [ant_list, pos_list]


def parse_yml_edgelist(yml_path: Path) -> list:
    """
    Parse edge list from YML file.

    Returns [anterior, posterior] where:
    - anterior = [L-A1, Acom, 3rd-A2, R-A1]
    - posterior = [L-Pcom, L-P1, R-P1, R-Pcom]
    """
    with open(yml_path, 'r') as f:
        data = yaml.safe_load(f)

    anterior = data['anterior']
    posterior = data['posterior']

    # Convert to edge lists in correct order
    ant_list = [anterior['L-A1'], anterior['Acom'], anterior['3rd-A2'], anterior['R-A1']]
    pos_list = [posterior['L-Pcom'], posterior['L-P1'], posterior['R-P1'], posterior['R-Pcom']]

    return [ant_list, pos_list]


def parse_edgelist(path: Path) -> list:
    """
    Dispatch to YAML or JSON parser based on file extension (case-insensitive).
    """
    suffix = path.suffix.lower()
    if suffix in ['.yml', '.yaml']:
        return parse_yml_edgelist(path)
    if suffix == '.json':
        return parse_json_edgelist(path)
    raise InvalidSubmissionError(
        f"Invalid edge-list extension {path.suffix}. Expected .yml, .yaml, or .json"
    )


def parse_json_edgelist(json_path: Path) -> list:
    """
    Parse edge list from JSON file (Grand Challenge format).

    Expected format:
    {
        "anterior": {"L-A1": 0/1, "Acom": 0/1, "3rd-A2": 0/1, "R-A1": 0/1},
        "posterior": {"L-Pcom": 0/1, "L-P1": 0/1, "R-P1": 0/1, "R-Pcom": 0/1}
    }

    Returns [anterior, posterior] where:
    - anterior = [L-A1, Acom, 3rd-A2, R-A1]
    - posterior = [L-Pcom, L-P1, R-P1, R-Pcom]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Validate format
    if not isinstance(data, dict):
        raise ValueError("JSON must be a dictionary")
    if set(data.keys()) != {"anterior", "posterior"}:
        raise ValueError(f"JSON must have 'anterior' and 'posterior' keys, got {data.keys()}")

    anterior = data['anterior']
    posterior = data['posterior']

    # Validate anterior keys
    if set(anterior.keys()) != {"L-A1", "Acom", "3rd-A2", "R-A1"}:
        raise ValueError(f"Anterior must have L-A1, Acom, 3rd-A2, R-A1 keys, got {anterior.keys()}")

    # Validate posterior keys
    if set(posterior.keys()) != {"L-Pcom", "L-P1", "R-P1", "R-Pcom"}:
        raise ValueError(f"Posterior must have L-Pcom, L-P1, R-P1, R-Pcom keys, got {posterior.keys()}")

    # Validate all values are 0 or 1
    all_values = list(anterior.values()) + list(posterior.values())
    if not all(value in [0, 1] for value in all_values):
        raise ValueError("All edge values must be 0 or 1")

    # Convert to edge lists in correct order
    ant_list = [anterior['L-A1'], anterior['Acom'], anterior['3rd-A2'], anterior['R-A1']]
    pos_list = [posterior['L-Pcom'], posterior['L-P1'], posterior['R-P1'], posterior['R-Pcom']]

    return [ant_list, pos_list]


def edge_list_to_variant_str(edge_list: list, region: str) -> str:
    """Convert edge list [0,1,0,1] to variant string."""
    assert region in ("Ant", "Pos"), "invalid region"
    edge_str = "_".join(map(str, edge_list))
    prefix = f"Var_{region}_"
    return prefix + edge_str


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


def aggregate_graph_dicts(all_graph_dicts: list) -> dict:
    """Aggregate graph dicts and compute variant-balanced accuracy."""
    list_ant_y_true = []
    list_ant_y_pred = []
    list_pos_y_true = []
    list_pos_y_pred = []

    for graph_dict in all_graph_dicts:
        list_ant_y_true.append(
            edge_list_to_variant_str(
                graph_dict["anterior"]["gt_graph"],
                region="Ant",
            )
        )
        list_ant_y_pred.append(
            edge_list_to_variant_str(
                graph_dict["anterior"]["pred_graph"],
                region="Ant",
            )
        )

        list_pos_y_true.append(
            edge_list_to_variant_str(
                graph_dict["posterior"]["gt_graph"],
                region="Pos",
            )
        )
        list_pos_y_pred.append(
            edge_list_to_variant_str(
                graph_dict["posterior"]["pred_graph"],
                region="Pos",
            )
        )

    ant_bal_acc = balanced_accuracy_score(list_ant_y_true, list_ant_y_pred)
    pos_bal_acc = balanced_accuracy_score(list_pos_y_true, list_pos_y_pred)

    return {
        "anterior": float(ant_bal_acc),
        "posterior": float(pos_bal_acc),
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame, submission_dir: Optional[Path] = None, answers_dir: Optional[Path] = None) -> dict:
    """
    Grade TopCoW Track 2 Task 3 submission using graph classification metrics.

    Predictions must be in YAML format (.yml or .yaml files).

    Returns all metrics from leaderboard:
    - anterior_accuracy, posterior_accuracy
    """
    if submission_dir is None:
        raise InvalidSubmissionError("Graph classification task requires submission_dir")

    if answers_dir is None:
        raise InvalidSubmissionError("Graph classification task requires answers_dir")

    # Validate submission format
    if 'image_id' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'image_id' column")
    if 'predicted_edges_path' not in submission.columns:
        raise InvalidSubmissionError("Submission must contain 'predicted_edges_path' column")

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
    all_graph_dicts = []

    for _, row in submission.iterrows():
        image_id = str(row['image_id'])

        # Normalize path - remove leading ./ and handle working/ directory
        pred_path_str = str(row['predicted_edges_path'])
        if pred_path_str.startswith('./'):
            pred_path_str = pred_path_str[2:]
        # Remove 'working/' prefix if present
        if pred_path_str.startswith('working/'):
            pred_path_str = pred_path_str[8:]

        pred_edges_path = submission_dir / pred_path_str

        # Find ground truth
        gt_row = answers[answers['image_id'].astype(str) == image_id]
        if gt_row.empty:
            raise InvalidSubmissionError(f"No ground truth found for {image_id}")

        gt_edges_path = answers_dir / gt_row.iloc[0]['label_path']

        # Validate files exist
        if not pred_edges_path.exists():
            raise InvalidSubmissionError(f"Predicted edges not found: {pred_edges_path}")
        if not gt_edges_path.exists():
            raise InvalidSubmissionError(f"Ground truth edges not found: {gt_edges_path}")

        try:
            # Load edge list from YAML or JSON (case-insensitive extensions)
            pred_ant_list, pred_pos_list = parse_edgelist(pred_edges_path)

            # Load ground truth edge list (GT files are YAML, but allow JSON for completeness)
            gt_ant_list, gt_pos_list = parse_edgelist(gt_edges_path)

            # Create graph dict
            graph_dict = {
                "anterior": {
                    "gt_graph": gt_ant_list,
                    "pred_graph": pred_ant_list,
                },
                "posterior": {
                    "gt_graph": gt_pos_list,
                    "pred_graph": pred_pos_list,
                },
            }

            all_graph_dicts.append(graph_dict)

        except Exception as e:
            raise InvalidSubmissionError(f"Error evaluating {image_id}: {e}")

    # Aggregate metrics
    if len(all_graph_dicts) == 0:
        raise InvalidSubmissionError("No metrics computed")

    accuracy_dict = aggregate_graph_dicts(all_graph_dicts)

    anterior_accuracy = accuracy_dict["anterior"]
    posterior_accuracy = accuracy_dict["posterior"]

    print(f"\n=== TopCoW Track 2 Task 3 Results ===")
    print(f"Cases evaluated: {len(all_graph_dicts)}")
    print(f"Anterior accuracy: {anterior_accuracy:.4f}")
    print(f"Posterior accuracy: {posterior_accuracy:.4f}")

    # Create results dict with metric names matching leaderboard
    results_dict = {
        "anterior_acc": anterior_accuracy,
        "posterior_acc": posterior_accuracy,
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
            results_dict['overall'] = -(anterior_accuracy + posterior_accuracy) / 2
    else:
        # Fall back to negative mean of metrics as overall score
        results_dict['overall'] = -(anterior_accuracy + posterior_accuracy) / 2

    return results_dict
