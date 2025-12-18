#!/usr/bin/env python3
"""
Script to fix and validate all challenge leaderboards.

This script:
1. Reads metric_config.json for each challenge to determine metric directions
2. Recalculates metric positions based on direction (higher_better or lower_better)
3. Recalculates mean_position as the average of all metric positions
4. Reorders teams by mean_position (ascending)
5. Updates rank numbers to reflect the new order

Usage:
    python fix_leaderboards.py
"""

import pandas as pd
import json
from pathlib import Path


def fix_leaderboard(challenge_dir):
    """
    Fix a single leaderboard based on metric_config.json.

    Args:
        challenge_dir: Path to challenge directory

    Returns:
        Tuple of (DataFrame or None, changed: bool, message: str)
    """
    challenge_path = Path(challenge_dir)
    leaderboard_path = challenge_path / 'leaderboard.csv'
    metric_config_path = challenge_path / 'metric_config.json'

    if not leaderboard_path.exists():
        return None, False, "No leaderboard.csv found"

    if not metric_config_path.exists():
        return None, False, "No metric_config.json found"

    # Read metric config
    try:
        with open(metric_config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        return None, False, f"Error reading metric_config.json: {e}"

    # Read leaderboard
    try:
        df = pd.read_csv(leaderboard_path)
    except Exception as e:
        return None, False, f"Error reading leaderboard.csv: {e}"

    original_df = df.copy()

    # Get metrics from config - handle both old and new formats
    if 'metric_directions' in config:
        metric_directions = config['metric_directions']
    elif 'metrics' in config:
        metric_directions = {m['name']: m['direction'] for m in config['metrics']}
    else:
        return None, False, "No metrics in config"

    metric_names = list(metric_directions.keys())
    if not metric_names:
        return None, False, "No metrics found"

    # Calculate positions for each metric
    for metric in metric_names:
        if metric not in df.columns:
            continue

        direction = metric_directions.get(metric, 'higher_better')
        if direction == 'lower_better':
            df[f'{metric}_position'] = df[metric].rank(ascending=True, method='min')
        else:  # higher_better
            df[f'{metric}_position'] = df[metric].rank(ascending=False, method='min')

    # Calculate mean_position
    position_cols = [f'{m}_position' for m in metric_names if f'{m}_position' in df.columns]
    if position_cols:
        df['mean_position'] = df[position_cols].mean(axis=1)
        # Round to 2 decimal places for consistency
        df['mean_position'] = df['mean_position'].round(2)

    # Reorder by mean_position
    df_sorted = df.sort_values('mean_position').reset_index(drop=True)
    df_sorted['rank'] = range(1, len(df_sorted) + 1)

    # Reorder columns: rank, team first, then metrics and positions
    cols = ['rank', 'team', 'mean_position']
    for metric in metric_names:
        if metric in df_sorted.columns:
            cols.append(metric)
        if f'{metric}_position' in df_sorted.columns:
            cols.append(f'{metric}_position')

    # Add any remaining columns that aren't in the ordered list
    for col in df_sorted.columns:
        if col not in cols:
            cols.append(col)

    df_sorted = df_sorted[cols]

    # Check if order changed (comparing team order)
    original_order = original_df['team'].tolist()
    new_order = df_sorted['team'].tolist()
    changed = original_order != new_order

    # Save
    try:
        df_sorted.to_csv(leaderboard_path, index=False)
    except Exception as e:
        return None, False, f"Error writing leaderboard.csv: {e}"

    return df_sorted, changed, "OK"


def main():
    """Main function to process all challenge leaderboards."""
    # Get all challenge directories
    base_path = Path(__file__).parent
    challenges = sorted([d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')])

    changes_summary = []
    errors_summary = []

    for challenge_dir in challenges:
        result, changed, message = fix_leaderboard(challenge_dir)

        if result is None:
            continue

        if "Error" in message or "error" in message:
            errors_summary.append((challenge_dir.name, message))
        else:
            status = "REORDERED" if changed else "no change"
            changes_summary.append((challenge_dir.name, status))

    # Print summary
    print("=" * 70)
    print("LEADERBOARD FIX SUMMARY")
    print("=" * 70)

    if changes_summary:
        print("\nProcessed challenges:")
        for challenge, status in sorted(changes_summary):
            symbol = "✓" if status == "REORDERED" else "-"
            print(f"{symbol} {challenge:40} {status}")

    if errors_summary:
        print("\nErrors encountered:")
        for challenge, error in errors_summary:
            print(f"✗ {challenge}: {error}")

    print("=" * 70)
    print(f"\nTotal challenges processed: {len(changes_summary)}")
    print(f"Challenges reordered: {sum(1 for _, s in changes_summary if s == 'REORDERED')}")
    if errors_summary:
        print(f"Errors: {len(errors_summary)}")


if __name__ == "__main__":
    main()
