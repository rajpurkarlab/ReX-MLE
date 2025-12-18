"""
Grading script for USenhance Challenge 2023

This script evaluates enhanced ultrasound images using three metrics:
- LNCC (Locally Normalized Cross-Correlation)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)

The final ranking is based on the average rank across all three metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import cv2
import json

from rexmle.grade_helpers import InvalidSubmissionError

# Load metric directions
CHALLENGE_DIR = Path(__file__).parent
with open(CHALLENGE_DIR / "metric_config.json") as f:
    METRIC_DIRECTIONS = json.load(f)["metric_directions"]


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image and convert to numpy array.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (grayscale)
    """
    try:
        # Try loading with PIL first
        img = Image.open(image_path)
        # Convert to grayscale if RGB
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.float32)
    except Exception as e:
        # Fallback to OpenCV
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img.astype(np.float32)


def calculate_lncc(img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> float:
    """
    Calculate Locally Normalized Cross-Correlation (LNCC).

    LNCC measures local similarity between two images by computing normalized
    cross-correlation in local windows.

    Args:
        img1: First image (predicted/enhanced)
        img2: Second image (ground truth)
        window_size: Size of local window (default: 11)

    Returns:
        LNCC score (higher is better, range: -1 to 1)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    # Normalize images to [0, 1]
    img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
    img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)

    # Compute local means
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    mu1 = cv2.filter2D(img1, -1, kernel)
    mu2 = cv2.filter2D(img2, -1, kernel)

    # Compute local variances and covariance
    mu1_sq = cv2.filter2D(img1 * img1, -1, kernel)
    mu2_sq = cv2.filter2D(img2 * img2, -1, kernel)
    mu1_mu2 = cv2.filter2D(img1 * img2, -1, kernel)

    sigma1_sq = mu1_sq - mu1 * mu1
    sigma2_sq = mu2_sq - mu2 * mu2
    sigma12 = mu1_mu2 - mu1 * mu2

    # Compute LNCC with proper handling of low-variance regions
    # Only include pixels where both images have sufficient local variance
    variance_threshold = 1e-10
    valid_mask = (sigma1_sq > variance_threshold) & (sigma2_sq > variance_threshold)

    if not np.any(valid_mask):
        # If no regions have variance, images are uniform - return 1.0 if identical, else 0.0
        return 1.0 if np.allclose(img1, img2) else 0.0

    # Compute LNCC only for valid regions
    numerator = sigma12[valid_mask]
    denominator = np.sqrt(sigma1_sq[valid_mask] * sigma2_sq[valid_mask])

    lncc_map = numerator / denominator
    lncc = np.mean(lncc_map)

    return float(lncc)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM).

    Args:
        img1: First image (predicted/enhanced)
        img2: Second image (ground truth)

    Returns:
        SSIM score (higher is better, range: -1 to 1, typically 0 to 1)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    # Use data_range for proper scaling
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())

    score = ssim(img1, img2, data_range=data_range)
    return float(score)


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


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        img1: First image (predicted/enhanced)
        img2: Second image (ground truth)

    Returns:
        PSNR score in dB (higher is better, inf for identical images)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    # Check if images are identical
    if np.array_equal(img1, img2):
        return float('inf')

    # Use data_range for proper scaling
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())

    score = psnr(img1, img2, data_range=data_range)
    return float(score)


def grade(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    submission_dir: Optional[Path] = None,
    answers_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Grade USenhance submissions using LNCC, SSIM, and PSNR metrics.

    The ranking method:
    1. Calculate LNCC, SSIM, and PSNR for each test image
    2. Rank submissions by each metric separately
    3. Compute average rank across the three metrics for final ranking

    Args:
        submission: Submission DataFrame with columns [image_id, enhanced_image_path]
        answers: Ground truth DataFrame with columns [image_id, ground_truth_path, private_path]
        submission_dir: Directory containing enhanced images
        answers_dir: Directory containing ground truth files

    Returns:
        Dictionary with metrics matching leaderboard columns:
        - mean_position: Placeholder for mean position (set to 0.0)
        - lncc: Average LNCC across all test images
        - lncc_position: Placeholder for LNCC rank position (set to 0.0)
        - ssim: Average SSIM across all test images
        - ssim_position: Placeholder for SSIM rank position (set to 0.0)
        - psnr: Average PSNR across all test images
        - psnr_position: Placeholder for PSNR rank position (set to 0.0)
        - overall: Combined score (sum of normalized metrics, for ranking)
    """

    # Validate submission format
    required_cols = ['image_id', 'enhanced_image_path']
    if not all(col in submission.columns for col in required_cols):
        raise InvalidSubmissionError(f"Submission must contain columns: {required_cols}")

    # Validate we have submission_dir
    if submission_dir is None:
        raise InvalidSubmissionError("submission_dir is required for image enhancement task")

    # Validate we have answers_dir
    if answers_dir is None:
        raise InvalidSubmissionError("answers_dir is required for image enhancement task")

    # Merge on image_id to align predictions with ground truth
    merged = answers.merge(submission, on='image_id', how='inner')

    # Check all test images have predictions
    if len(merged) != len(answers):
        missing = set(answers['image_id']) - set(submission['image_id'])
        raise InvalidSubmissionError(
            f"Missing predictions for {len(missing)} images. "
            f"Expected {len(answers)}, got {len(submission)}. "
            f"Missing IDs: {list(missing)[:10]}"
        )

    # Calculate metrics for each image
    lncc_scores = []
    ssim_scores = []
    psnr_scores = []

    for idx, row in merged.iterrows():
        image_id = row['image_id']

        # Load enhanced image (submission)
        # Handle both cases: path is relative to submission_dir, or path includes submission_dir name
        enhanced_rel_path = row['enhanced_image_path']
        enhanced_path = submission_dir / enhanced_rel_path

        # If the path doesn't exist, try without the first directory component
        # (handles case where CSV says "enhanced/file.png" but submission_dir is already ".../enhanced")
        if not enhanced_path.exists():
            if '/' in enhanced_rel_path or '\\' in enhanced_rel_path:
                # Try removing first path component
                parts = Path(enhanced_rel_path).parts
                if len(parts) > 1:
                    enhanced_path_alt = submission_dir / Path(*parts[1:])
                    if enhanced_path_alt.exists():
                        enhanced_path = enhanced_path_alt

        if not enhanced_path.exists():
            raise InvalidSubmissionError(
                f"Enhanced image not found: {enhanced_path}\n"
                f"  Submission dir: {submission_dir}\n"
                f"  Relative path: {enhanced_rel_path}"
            )

        # Load ground truth image from answers_dir
        gt_path = answers_dir / row['ground_truth_path']
        if not gt_path.exists():
            raise InvalidSubmissionError(f"Ground truth image not found: {gt_path}")

        try:
            enhanced_img = load_image(enhanced_path)
            gt_img = load_image(gt_path)

            # Resize enhanced image to match ground truth if needed
            if enhanced_img.shape != gt_img.shape:
                enhanced_img = cv2.resize(
                    enhanced_img,
                    (gt_img.shape[1], gt_img.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            # Calculate metrics
            lncc = calculate_lncc(enhanced_img, gt_img)
            ssim_score = calculate_ssim(enhanced_img, gt_img)
            psnr_score = calculate_psnr(enhanced_img, gt_img)

            lncc_scores.append(lncc)
            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)

        except Exception as e:
            raise InvalidSubmissionError(
                f"Error processing image {image_id}: {str(e)}"
            )

    # Calculate average metrics
    avg_lncc = np.mean(lncc_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)

    print(f"\n=== USEnhance Results ===")
    print(f"Images evaluated: {len(lncc_scores)}")
    print(f"LNCC: {avg_lncc:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"PSNR: {avg_psnr:.4f}")

    # Create results dict
    results_dict = {
        'lncc': float(avg_lncc),
        'ssim': float(avg_ssim),
        'psnr': float(avg_psnr),
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
            results_dict['overall'] = -(avg_lncc + avg_ssim + avg_psnr / 30)
    else:
        # Fall back to negative mean of metrics as overall score
        results_dict['overall'] = -(avg_lncc + avg_ssim + avg_psnr / 30)

    return results_dict
