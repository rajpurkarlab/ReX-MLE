import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Dict


def grade(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    **kwargs
) -> Dict[str, float]:
    """
    Grade LDCT-IQA predictions using correlation metrics.

    Calculates PLCC, SROCC, and KROCC between predicted quality scores
    and ground truth radiologist scores.

    Args:
        submission: Submission DataFrame with columns [image_id, quality_score]
        answers: Ground truth DataFrame with columns [image_id, quality_score]

    Returns:
        Dictionary with metrics:
        - plcc: Pearson Linear Correlation Coefficient
        - srocc: Spearman Rank Order Correlation Coefficient
        - krocc: Kendall Rank Order Correlation Coefficient
        - overall: PLCC + SROCC + KROCC (primary leaderboard metric)
    """

    # Validate submission format
    required_cols = ['image_id', 'quality_score']
    if not all(col in submission.columns for col in required_cols):
        raise ValueError(f"Submission must contain columns: {required_cols}")

    # Merge on image_id to align predictions with ground truth
    merged = answers.merge(submission, on='image_id', suffixes=('_gt', '_pred'))

    # Check all test images have predictions
    if len(merged) != len(answers):
        missing = set(answers['image_id']) - set(submission['image_id'])
        raise ValueError(f"Missing predictions for {len(missing)} images: {list(missing)[:10]}")

    # Extract prediction and ground truth arrays
    total_pred = merged['quality_score_pred'].values
    total_gt = merged['quality_score_gt'].values

    # Calculate correlation metrics (using absolute values as per challenge spec)
    plcc = abs(pearsonr(total_pred, total_gt)[0])
    srocc = abs(spearmanr(total_pred, total_gt)[0])
    krocc = abs(kendalltau(total_pred, total_gt)[0])

    # Return dictionary with all metrics
    score_sum = plcc + srocc + krocc
    results_dict = {
        "plcc": plcc,
        "srocc": srocc,
        "krocc": krocc,
        "score": score_sum,  # Add score to match leaderboard column
        "overall": score_sum
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
    "plcc": "higher_better",
    "srocc": "higher_better",
    "krocc": "higher_better",
    "score": "higher_better"
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
