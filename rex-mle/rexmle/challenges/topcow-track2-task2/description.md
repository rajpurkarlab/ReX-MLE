# TopCoW Track 2 Task 2 - MRA 3D Bounding Box Detection

**Challenge Website:** https://topcow24.grand-challenge.org/
**Year:** 2024
**Conference:** MICCAI 2024
**Track:** 2 (MRA)
**Task:** 2 (3D Bounding Box Detection)
**Data License:** https://opendata.swiss/en/terms-of-use

## Overview

TopCoW Track 2 Task 2 focuses on detecting the Circle of Willis region using 3D bounding boxes on MRA (Magnetic Resonance Angiography) imaging. The goal is to accurately localize the CoW vascular structure.

## Task Description

**Objective:** Detect the Circle of Willis region with a 3D bounding box on MRA images.

**Input:** 3D MRA volume in NIfTI format
**Output:** JSON file with bounding box coordinates (`{"size": [x,y,z], "location": [x,y,z]}`)

## Dataset

### Training Data
- **125 MRA cases** with 3D bounding box annotations
- **Format:** NIfTI (.nii.gz), 16-bit signed, LPS+ orientation
- **Naming:** `topcow_mr_{pat_id}_0000.nii.gz`

### Test Data
- **~25 MRA test cases** (20% split)
- Images provided, labels withheld for evaluation

### Annotation Format
The training labels are provided as text files containing bounding box coordinates.

## Evaluation Metrics

### Primary Metric
- **3D IoU (Intersection over Union)**: Overlap between predicted and ground truth bounding boxes

### Secondary Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

## Submission Format

Your submission must include:

1. **submission.csv** with columns:
   - `image_id`: Patient ID (e.g., "001")
   - `modality`: "MRA"
   - `predicted_bbox_path`: Path to prediction JSON file (e.g., "predictions/topcow_mr_001_bbox.json")

2. **predictions/** directory with JSON files:
   - Each JSON file contains bounding box coordinates
   - Format: `{"size": [x, y, z], "location": [x, y, z]}`
   - `size`: Dimensions of the bounding box in voxels [width, height, depth]
   - `location`: Center coordinates of the bounding box in voxel space [x, y, z]

Example:
```
submission/
├── submission.csv
└── predictions/
    ├── topcow_mr_001_bbox.json
    ├── topcow_mr_002_bbox.json
    └── ...
```

Example JSON file content:
```json
{
  "size": [128, 128, 64],
  "location": [256, 256, 128]
}
```

## References

- Challenge Website: https://topcow24.grand-challenge.org/
- Zenodo Dataset: https://zenodo.org/records/15692630
- License: https://opendata.swiss/en/terms-of-use

## Citation

If you use this dataset, please cite the TopCoW 2024 challenge.
