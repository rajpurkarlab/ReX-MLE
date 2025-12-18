# TopCoW Track 1 Task 3 - CTA Graph Edge Classification

**Challenge Website:** https://topcow24.grand-challenge.org/
**Year:** 2024
**Conference:** MICCAI 2024
**Track:** 1 (CTA)
**Task:** 3 (Graph Edge Classification)
**Data License:** https://opendata.swiss/en/terms-of-use

## Overview

TopCoW Track 1 Task 3 focuses on classifying topological graph edges in the Circle of Willis on CTA (Computed Tomography Angiography) imaging. The goal is to accurately classify different edge types in the CoW graph structure.

## Task Description

**Objective:** Classify topological edges in the Circle of Willis graph on CTA images.

**Input:** 3D CTA volume in NIfTI format
**Output:** JSON file with anterior/posterior edge classifications

## Dataset

### Training Data
- **125 CTA cases** with graph edge classification annotations
- **Format:** NIfTI (.nii.gz), 16-bit signed, LPS+ orientation
- **Naming:** `topcow_ct_{pat_id}_0000.nii.gz`

### Test Data
- **~25 CTA test cases** (20% split)
- Images provided, labels withheld for evaluation

### Edge Classification
The task classifies the presence (1) or absence (0) of specific topological edges:
- **Anterior edges**: L-A1, Acom (Anterior communicating artery), 3rd-A2, R-A1
- **Posterior edges**: L-Pcom, L-P1, R-P1, R-Pcom

## Evaluation Metrics

### Primary Metrics
- **Anterior Accuracy**: Variant-balanced accuracy for anterior topology classification
- **Posterior Accuracy**: Variant-balanced accuracy for posterior topology classification

The evaluation uses balanced accuracy to account for anatomical variant imbalance in the dataset.

## Submission Format

Your submission must include:

1. **submission.csv** with columns:
   - `image_id`: Patient ID (e.g., "001")
   - `modality`: "CTA"
   - `predicted_edges_path`: Path to prediction JSON file (e.g., "predictions/topcow_ct_001_edges.json")

2. **predictions/** directory with JSON files:
   - Each JSON file contains edge classifications
   - Format: Dictionary with "anterior" and "posterior" keys
   - Each edge is marked as present (1) or absent (0)

Example:
```
submission/
├── submission.csv
└── predictions/
    ├── topcow_ct_001_edges.json
    ├── topcow_ct_002_edges.json
    └── ...
```

Example JSON file content:
```json
{
  "anterior": {
    "L-A1": 1,
    "Acom": 1,
    "3rd-A2": 0,
    "R-A1": 1
  },
  "posterior": {
    "L-Pcom": 0,
    "L-P1": 1,
    "R-P1": 1,
    "R-Pcom": 0
  }
}
```

## References

- Challenge Website: https://topcow24.grand-challenge.org/
- Zenodo Dataset: https://zenodo.org/records/15692630
- License: https://opendata.swiss/en/terms-of-use

## Citation

If you use this dataset, please cite the TopCoW 2024 challenge.
