# TopBrain Track 1 - CTA Multiclass Brain Vessel Segmentation

**Challenge Website:** https://topbrain2025.grand-challenge.org/
**Year:** 2025
**Conference:** MICCAI 2025
**Track:** 1 (CTA)
**Data License:** https://opendata.swiss/en/terms-of-use

## Overview

TopBrain Track 1 focuses on multiclass segmentation of brain vessels using CTA (Computed Tomography Angiography) imaging. The challenge emphasizes topology-aware evaluation to ensure anatomically accurate vessel segmentation beyond voxel-wise accuracy.

## Task Description

**Objective:** Segment individual brain vessels into 40 anatomical classes on CTA images with topologically accurate connectivity.

**Input:** 3D CTA volume in NIfTI format
**Output:** Multiclass segmentation mask in NIfTI format with 40 vessel labels

## Dataset

### Training Data
- **25 CTA training patients** with paired CTA and MRA data
- **40 vessel classes** for CTA segmentation
- **Format:** NIfTI (.nii.gz), 16-bit signed, LPS+ orientation
- **Naming:** `topcow_ct_{pat_id}_0000.nii.gz`
- **Labels:** `topcow_ct_{pat_id}_0000.nii.gz` in `labelsTr_topbrain_ct/`

### Test Data
- **20 test patients** (exact split: 80/20 with random_state=42)
- Images provided, labels withheld for evaluation

### Vessel Classes (40 total)
The segmentation includes 40 vessel classes representing different segments of cerebral vessels:
- Background (class 0)
- Individual brain vessel segments (classes 1-40)
- Each class represents a specific anatomical vessel segment

## Evaluation Metrics

### Primary Metric
- **Mean Dice Similarity Coefficient**: Average Dice score across all 40 vessel classes

### Secondary Metrics
- **Mean IoU**: Intersection over Union averaged across classes
- **Topology Metrics**:
  - Centerline accuracy
  - Connected components analysis
  - Neighborhood topology preservation

### Topology-Focused Evaluation
This challenge emphasizes topologically correct segmentations:
- Vessel centerline extraction and matching
- Connected component analysis per vessel class
- Neighborhood relationships between vessels
- Anatomical connectivity validation

## Submission Format

Your submission must include:

1. **submission.csv** with columns:
   - `image_id`: Patient ID (e.g., "001")
   - `modality`: "CTA"
   - `predicted_mask_path`: Path to prediction (e.g., "predictions/topcow_ct_001_0000.nii.gz")

2. **predictions/** directory:
   - Multiclass segmentation masks in NIfTI format
   - Same dimensions and spacing as input images
   - Integer values 0-40 representing different vessel classes
   - Background = 0, vessels = 1-40

Example:
```
submission/
├── submission.csv
└── predictions/
    ├── topcow_ct_001_0000.nii.gz
    ├── topcow_ct_002_0000.nii.gz
    └── ...
```

## Data Structure

The dataset follows this structure:
```
imagesTr_topbrain_ct/      # CTA training images
├── topcow_ct_001_0000.nii.gz
├── topcow_ct_002_0000.nii.gz
└── ...

labelsTr_topbrain_ct/      # CTA segmentation labels (40 classes)
├── topcow_ct_001_0000.nii.gz
├── topcow_ct_002_0000.nii.gz
└── ...
```

## Technical Details

### Image Properties
- **Modality**: CTA (Computed Tomography Angiography)
- **Format**: NIfTI (.nii.gz)
- **Orientation**: LPS+ (Left-Posterior-Superior)
- **Data Type**: 16-bit signed integer
- **Dimensions**: Variable per patient (3D volumes)

### Label Properties
- **Number of Classes**: 41 (including background)
- **Label Range**: 0-40
- **Background**: Class 0
- **Vessel Classes**: 1-40

## References

- Challenge Website: https://topbrain.grand-challenge.org/
- Zenodo Dataset: https://zenodo.org/records/16878417
- License: https://opendata.swiss/en/terms-of-use

## Citation

If you use this dataset, please cite the TopBrain 2025 challenge.

## Related Challenges

- **TopBrain Track 2**: MRA multiclass brain vessel segmentation with 42 vessel classes
- **TopCoW 2024**: Predecessor challenge focusing on Circle of Willis
