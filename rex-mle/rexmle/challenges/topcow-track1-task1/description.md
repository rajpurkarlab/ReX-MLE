# TopCoW Track 1 Task 1 - CTA Multi-class Segmentation

**Challenge Website:** https://topcow24.grand-challenge.org/
**Year:** 2024
**Conference:** MICCAI 2024
**Track:** 1 (CTA)
**Task:** 1 (Multi-class Segmentation)
**Data License:** https://opendata.swiss/en/terms-of-use

## Overview

TopCoW Track 1 Task 1 focuses on multi-class segmentation of the Circle of Willis vessels using CTA (Computed Tomography Angiography) imaging. The goal is to accurately segment individual vessels of this critical cerebrovascular structure.

## Task Description

**Objective:** Segment individual vessels of the Circle of Willis into multiple anatomical classes on CTA images.

**Input:** 3D CTA volume in NIfTI format
**Output:** Multi-class segmentation mask in NIfTI format

## Dataset

### Training Data
- **125 CTA cases** with multi-class segmentation annotations
- **Format:** NIfTI (.nii.gz), 16-bit signed, LPS+ orientation
- **Naming:** `topcow_ct_{pat_id}_0000.nii.gz`

### Test Data
- **~25 CTA test cases** (20% split)
- Images provided, labels withheld for evaluation

### Vessel Classes
The segmentation includes multiple vessel classes representing different segments of the Circle of Willis:
- Background (class 0)
- Individual CoW vessels (classes 1-N)

## Evaluation Metrics

### Primary Metric
- **Mean Dice Similarity Coefficient**: Average Dice score across all vessel classes

### Secondary Metrics
- **Mean IoU**: Intersection over Union averaged across classes
- **Surface Dice** (optional): Surface-based accuracy metric

## Submission Format

Your submission must include:

1. **submission.csv** with columns:
   - `image_id`: Patient ID (e.g., "001")
   - `modality`: "CTA"
   - `predicted_mask_path`: Path to prediction (e.g., "predictions/topcow_ct_001_0000.nii.gz")

2. **predictions/** directory:
   - Multi-class segmentation masks in NIfTI format
   - Same dimensions and spacing as input images
   - Integer values representing different vessel classes

Example:
```
submission/
├── submission.csv
└── predictions/
    ├── topcow_ct_001_0000.nii.gz
    ├── topcow_ct_002_0000.nii.gz
    └── ...
```

## References

- Challenge Website: https://topcow24.grand-challenge.org/
- Zenodo Dataset: https://zenodo.org/records/15692630
- License: https://opendata.swiss/en/terms-of-use

## Citation

If you use this dataset, please cite the TopCoW 2024 challenge.
