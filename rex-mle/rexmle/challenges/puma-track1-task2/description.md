# PUMA Track 1 Task 2 - Nuclei Detection (3 Classes)

**Challenge Website:** https://puma.grand-challenge.org/
**Year:** 2024
**Track:** 1 (Three Instance Classes)
**Task:** 2 (Nuclei Detection)
**Dataset:** https://zenodo.org/records/14869398

## Overview

PUMA Track 1 Task 2 focuses on nuclei detection and classification in melanoma histopathology images using a simplified three-class system. This task aims to detect and classify nuclei into tumor cells, tumor-infiltrating lymphocytes (TILs), and other cell types, which is crucial for understanding the tumor microenvironment and predicting treatment response.

## Task Description

**Objective:** Detect and classify individual nuclei into three broad categories: tumor, TILs (lymphocytes + plasma cells), and other cells.

**Input:** H&E stained histopathology images (TIF format, 1024×1024 pixels at 40× magnification)
**Output:** List of detected nuclei with centroid coordinates and class labels

## Dataset

### Training Data
- **204 cases** (103 primary melanoma + 102 metastatic melanoma, with one excluded case)
- **Format:** TIF images, 1024×1024 pixels
- **Resolution:** 40× magnification (0.23 μm per pixel)
- **Annotations:** GeoJSON format with nuclei instance segmentation and class labels
- **Annotation Quality:** All annotations reviewed and verified by a board-certified dermatopathologist

### Test Data
- **~41 test cases** (20% split)
- Images provided, labels withheld for evaluation

### Nuclei Classes (3-Class System)

1. **Tumor**: Melanoma tumor cell nuclei
2. **TILs**: Tumor-infiltrating lymphocytes
   - Includes: Lymphocytes and plasma cells
3. **Other**: All other cell types
   - Includes: Histiocytes, melanophages, neutrophils, stromal cells, epithelial cells, vascular endothelium, and apoptotic cells

### Annotation Process

The nuclei annotations were generated using:
1. **Initial segmentation**: Hover-Net pretrained on PanNuke dataset
2. **Manual refinement**: Performed by trained medical expert (M.S.) using QuPath
3. **Expert validation**: All annotations reviewed and corrected by board-certified dermatopathologist (W.B.)

### Quality Metrics

Validation on 12 randomly selected ROIs:

**Intraobserver agreement:**
- Precision: 84.89%
- Recall: 86.45%
- F1 Score: 85.66%

**Interobserver agreement:**
- Precision: 80.34%
- Recall: 80.62%
- F1 Score: 80.20%

## Clinical Motivation

Accurate nuclei detection and classification enables:
- **TIL quantification**: Measure tumor-infiltrating lymphocytes for immunotherapy response prediction
- **Spatial analysis**: Understand tumor-immune interactions
- **Prognostic biomarkers**: Develop predictive models for treatment outcomes

Approximately 50% of melanoma patients do not respond to immune checkpoint inhibition therapy, making accurate biomarker development critical.

## Evaluation Metrics

### Primary Metric
- **Macro F1 Score**: Average F1 score across the 3 nuclei classes (tumor, TILs, other)
  - Follows official PUMA Grand Challenge evaluation methodology
  - Uses centroid-based matching with 15-pixel distance threshold
  - Predictions matched by highest confidence score, then nearest distance
  - One-to-one matching (each prediction and ground truth matched at most once)

### Evaluation Process
1. Extract centroids from polygon annotations
2. For each ground truth nucleus, find predictions within 15-pixel radius
3. Sort eligible predictions by confidence (descending) then distance (ascending)
4. Match best prediction to ground truth
5. Count TP/FP/FN per class
6. Compute F1 per class, then average (macro)

### Secondary Metrics
- **Per-class F1 scores**: Individual F1 scores for tumor, TILs, and other cells
- **Micro F1 Score**: F1 computed across all nuclei regardless of class
- **Precision and Recall**: Per-class detection metrics

## Submission Format

Your submission must include:

1. **submission.csv** with columns:
   - `case_id`: Case identifier (e.g., "training_set_primary_roi_001")
   - `predicted_nuclei_path`: Relative path to prediction JSON file

2. **predictions/** directory with nuclei detection results:
   - **Format**: JSON files in "Multiple Polygons" format (matches Grand Challenge)
   - **Classes**: `tumor`, `TILs`, or `other`
   - **Coordinates**: Polygon path points (centroids computed automatically)
   - **Confidence scores**: Optional but recommended (default: 1.0)

**Required JSON format (Multiple Polygons):**
```json
{
  "polygons": [
    {
      "name": "tumor",
      "path_points": [[512, 256], [513, 256], [513, 257], ...],
      "score": 0.95
    },
    {
      "name": "TILs",
      "path_points": [[128, 384], [129, 384], [129, 385], ...],
      "score": 0.89
    },
    {
      "name": "other",
      "path_points": [[200, 100], [201, 100], [201, 101], ...],
      "score": 0.75
    }
  ]
}
```

**Simplified format (also supported):**
```json
{
  "nuclei": [
    {
      "centroid": [512, 256],
      "class": "tumor",
      "confidence": 0.95
    }
  ]
}
```

**Example submission structure:**
```
submission/
├── submission.csv
└── predictions/
    ├── training_set_primary_roi_001.json
    ├── training_set_metastatic_roi_045.json
    └── ...
```

**Note:** The evaluation uses centroid-based matching. If you provide polygons, centroids are computed automatically by averaging path_points. Including confidence scores improves matching accuracy.

Example:
```
submission/
├── submission.csv
└── predictions/
    ├── training_set_primary_roi_001.json
    ├── training_set_metastatic_roi_045.json
    └── ...
```

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
