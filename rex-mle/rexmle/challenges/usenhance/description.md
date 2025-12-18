# Ultrasound Image Enhancement Challenge 2023

**Grand Challenge:** https://ultrasoundenhance2023.grand-challenge.org/ultrasoundenhance2023/

## Overview

The USenhance Challenge focuses on enhancing ultrasound images acquired from handheld devices to match the quality of images from high-end hospital machines. This challenge addresses a critical barrier in expanding ultrasound accessibility: low imaging quality from portable devices due to hardware limitations.

Ultrasound imaging is widely used for disease diagnosis and treatment due to its non-invasive nature. Recently, medical ultrasound has evolved from expensive, large hospital machines to economical handheld devices with broader applications. However, handheld ultrasound devices suffer from lower image quality due to hardware constraints. Image enhancement algorithms provide a cost-effective solution by computationally improving image quality without requiring hardware improvements, enabling wider adoption of portable ultrasound technology.

## Task Description

### Clinical Context
Portable ultrasound devices enable point-of-care imaging in emergency rooms, ambulances, remote areas, and developing countries. However, reduced image quality can compromise diagnostic accuracy. This challenge aims to develop algorithms that can enhance low-quality ultrasound images to clinical diagnostic standards.

### Imaging Details
- **Modality**: Ultrasound (B-mode)
- **Organs**: Five different anatomical regions
  - Thyroid
  - Carotid artery
  - Liver
  - Breast
  - Kidney
- **Patients**: 109 patients total
- **Image Pairs**: 1500 paired low-quality and high-quality images (3000 images total)
- **Task**: Reconstruct high-quality images from low-quality handheld device images

### Data Characteristics
- **Low-quality images**: Acquired from handheld ultrasound devices with hardware limitations
- **High-quality images**: Reference images from professional hospital-grade ultrasound machines
- **Format**: Standard image formats (PNG/JPEG)
- **Paired data**: Each low-quality image has a corresponding high-quality reference

## Dataset Split (Med-MLE-Bench)

The original training dataset (1500 pairs) is split into:
- **Training Set**: 80% (1200 pairs) - Both low and high-quality images provided
- **Test Set**: 20% (300 pairs) - Only low-quality images provided publicly, high-quality images held for evaluation

Split is deterministic with `random_state=42` to ensure reproducibility across users.

**Note**: The original challenge test set contains only low-quality images without high-quality references, making it unsuitable for automated evaluation. Therefore, we create our test set from the training data.

### Data Structure

```
public/
├── train/
│   ├── low_quality/    # Input images from handheld devices
│   │   ├── image_001.png
│   │   └── ...
│   └── high_quality/   # Target reference images
│       ├── image_001.png
│       └── ...
├── test/
│   └── low_quality/    # Test images to enhance
│       ├── image_100.png
│       └── ...
└── sample_submission.csv
```

## Goals

1. **Enhance Image Quality**: Improve visual quality and diagnostic utility of handheld ultrasound images
2. **Multi-organ Generalization**: Develop methods that work across different anatomical regions
3. **Preserve Diagnostic Features**: Maintain clinical relevance while enhancing image quality
4. **Enable Clinical Translation**: Create practical solutions for real-world portable ultrasound applications

## Evaluation Metrics

The challenge uses three widely-used image quality assessment metrics:

### 1. LNCC (Locally Normalized Cross-Correlation)
Measures local similarity between enhanced and reference images using normalized cross-correlation in local windows. Higher values indicate better correlation.
- **Range**: -1 to 1
- **Higher is better**

### 2. SSIM (Structural Similarity Index)
Evaluates structural similarity between images, considering luminance, contrast, and structure. Widely used for perceptual image quality assessment.
- **Range**: 0 to 1
- **Higher is better**

### 3. PSNR (Peak Signal-to-Noise Ratio)
Measures pixel-level similarity between enhanced and reference images in decibels.
- **Typical range**: 20-50 dB
- **Higher is better**

### Ranking Method

For each test image:
1. Calculate LNCC, SSIM, and PSNR
2. Rank all submissions by each metric separately
3. Compute the **mean position** (average rank) across the three metrics
4. **Final ranking** is based on the mean position (lower is better)

Example from leaderboard:
- Team POAS: Mean Position = 1.3 (LNCC rank=1, SSIM rank=2, PSNR rank=1)
- Team NCIA: Mean Position = 1.7 (LNCC rank=2, SSIM rank=1, PSNR rank=2)

### Overall Score (Med-MLE-Bench)
For automated evaluation, we provide an overall score that combines normalized metrics:
```python
normalized_lncc = (lncc + 1) / 2  # Map [-1,1] to [0,1]
normalized_ssim = ssim             # Already in [0,1]
normalized_psnr = (psnr - 20) / 30 # Map typical range to [0,1]
overall = (normalized_lncc + normalized_ssim + normalized_psnr) / 3
```

## Submission Format

Your submission should include:

1. **CSV file** (`submission.csv`) with columns:
   - `image_id`: Image identifier (without extension)
   - `enhanced_image_path`: Relative path to enhanced image

2. **Enhanced images**: Directory containing all enhanced images

Example CSV:
```csv
image_id,enhanced_image_path
image_001,enhanced/image_001.png
image_002,enhanced/image_002.png
image_003,enhanced/image_003.png
...
```

Enhanced images should:
- Match the dimensions of the corresponding low-quality input (or will be resized during evaluation)
- Be in standard image format (PNG, JPEG)
- Preserve anatomical structures and diagnostic features

## Challenge Information

- **Year**: 2023
- **Conference**: MICCAI 2023
- **Workshop**: USenhance Workshop
- **Meeting Recording**: [MICCAI2023] USenhance Workshop - YouTube
- **Online Meeting Date**: October 12, 2023
- **Grand Challenge Link**: https://ultrasoundenhance2023.grand-challenge.org/ultrasoundenhance2023/

## Leaderboard Highlights

Top performers from the original challenge:

| Rank | Team | Mean Position | LNCC | SSIM | PSNR |
|------|------|---------------|------|------|------|
| 1st | POAS (ych000) | 1.3 | 0.9080 | 0.7439 | 30.7268 |
| 2nd | NCIA (gkdl3000) | 1.7 | 0.9065 | 0.7576 | 30.4643 |
| 3rd | imed_8 | 6.0 | 0.9012 | 0.7407 | 29.5355 |

See `leaderboard.csv` for complete results.

## Data Source

- **Grand Challenge**: https://ultrasoundenhance2023.grand-challenge.org/ultrasoundenhance2023/
- **Training Data**: Google Drive (1500 paired images)
- **Original Dataset Size**: 3000 images (1500 pairs) from 109 patients

## Additional Resources

- Workshop recording: [MICCAI2023] USenhance Workshop - YouTube
- Challenge website: https://ultrasoundenhance2023.grand-challenge.org/ultrasoundenhance2023/
