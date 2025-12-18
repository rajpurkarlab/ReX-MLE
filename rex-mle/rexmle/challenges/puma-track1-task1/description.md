# PUMA Track 1 Task 1 - Semantic Tissue Segmentation

**Challenge Website:** https://puma.grand-challenge.org/
**Year:** 2024
**Track:** 1 (Three Instance Classes)
**Task:** 1 (Semantic Tissue Segmentation)
**Dataset:** https://zenodo.org/records/14869398

## Overview

PUMA Track 1 Task 1 focuses on semantic tissue segmentation in melanoma histopathology images. The goal is to accurately segment different tissue types in H&E stained whole slide images to support the development of prognostic biomarkers for melanoma treatment response prediction.

## Task Description

**Objective:** Segment tissue regions into five semantic classes: tumor, stroma, epithelium, blood vessel, and necrotic regions.

**Input:** H&E stained histopathology images (TIF format, 1024×1024 pixels at 40× magnification)
**Output:** Semantic segmentation mask with tissue class labels

## Dataset

### Training Data
- **204 cases** (103 primary melanoma + 102 metastatic melanoma, with one excluded case)
- **Format:** TIF images, 1024×1024 pixels
- **Resolution:** 40× magnification (0.23 μm per pixel)
- **Annotations:** GeoJSON format with tissue class labels
- **Annotation Quality:** All annotations reviewed and verified by a board-certified dermatopathologist

### Context ROIs
- Additional 5120×5120 pixel context ROIs available for understanding broader tissue context

### Test Data
- **~41 test cases** (20% split)
- Images provided, labels withheld for evaluation

### Tissue Classes
1. **Tumor**: Melanoma tumor cells
2. **Stroma**: Connective tissue supporting structures
3. **Epithelium**: Epithelial tissue (e.g., epidermis)
4. **Blood Vessel**: Vascular structures
5. **Necrosis**: Necrotic/dead tissue regions

### Quality Metrics
The dataset annotations have been validated with:
- **Intraobserver DICE score**: 0.90 (average across 12 ROIs)
- **Interobserver DICE score**: 0.90 (agreement with second pathologist)

## Clinical Motivation

Advanced melanoma treatment with immune checkpoint inhibition (ICI) is costly and potentially toxic, yet approximately half of patients do not respond to therapy. Accurate tissue segmentation enables:
- Localization of tumor-infiltrating lymphocytes (TILs)
- Assessment of tumor-stroma interactions
- Development of spatial biomarkers for treatment response prediction

## Evaluation Metrics

### Primary Metric
- **Micro Dice Score**: Concatenates all predicted and ground truth masks, computes Dice score per tissue class, then averages across classes
  - Follows official PUMA Grand Challenge evaluation methodology
  - Background (class 0) excluded from evaluation
  - Standard image size: 1024×1024 pixels

### Secondary Metrics
- **Macro Dice Score**: Average Dice score per case across all test cases
- **Per-class Dice scores**: Individual Dice scores for each tissue class (stroma, blood vessel, tumor, epidermis, necrosis)

## Submission Format

Your submission must include:

1. **submission.csv** with columns:
   - `case_id`: Case identifier (e.g., "training_set_primary_roi_001")
   - `predicted_mask_path`: Relative path to prediction mask file

2. **predictions/** directory with segmentation masks:
   - **Format**: PNG or TIF files
   - **Dimensions**: 1024×1024 pixels (same as input images)
   - **Pixel values**: Integer labels representing tissue classes:
     - `0`: Background (white/empty regions)
     - `1`: Stroma (connective tissue)
     - `2`: Blood Vessel
     - `3`: Tumor (melanoma cells)
     - `4`: Epidermis (epithelial tissue)
     - `5`: Necrosis (dead tissue)

**Example submission structure:**
```
submission/
├── submission.csv
└── predictions/
    ├── training_set_primary_roi_001.tif
    ├── training_set_metastatic_roi_045.tif
    └── ...
```

**Example submission.csv:**
```csv
case_id,predicted_mask_path
training_set_primary_roi_001,predictions/training_set_primary_roi_001.tif
training_set_metastatic_roi_045,predictions/training_set_metastatic_roi_045.tif
```

**Note:** This format matches the real PUMA Grand Challenge evaluation where ground truth is stored as TIF masks and predictions are evaluated using the Micro Dice metric.

## References

- Challenge Website: https://puma.grand-challenge.org/
- Zenodo Dataset: https://zenodo.org/records/14869398
- Scanner: Hamamatsu, 40× magnification (0.23 μm per pixel)
- Annotation Tool: QuPath
- Initial Segmentation: Hover-Net pretrained on PanNuke dataset

## Citation

If you use this dataset, please cite:

Schuiveling, Mark; Blokx, Willeke; Breimer, Gerben (2024):
Melanoma Histopathology Dataset with Tissue and Nuclei Annotations.
Zenodo. https://doi.org/10.5281/zenodo.14869398

## Dataset Versions

- **Version 3**: Removed sample "training_set_metastatic_roi_103" due to annotation inconsistencies
- **Version 4** (Current): Fixed color annotation issue in training_set_metastatic_roi_088
