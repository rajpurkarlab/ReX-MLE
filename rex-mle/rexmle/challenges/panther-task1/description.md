# PANTHER Task 1: Pancreatic Tumor Segmentation on Diagnostic MRI

**Grand Challenge:** https://panther.grand-challenge.org/

## Overview

PANTHER Task 1 focuses on pancreatic tumor segmentation from T1-weighted contrast-enhanced diagnostic MRI scans. Pancreatic cancer has a critically low 5-year survival rate (10-12%), making accurate detection and treatment planning essential.

This task provides the **first publicly available dataset** for pancreatic diagnostic MRI segmentation, aiming to bridge the gap between research and clinical practice by fostering AI solutions for pancreatic cancer imaging.

## Task Description

### Clinical Context
Diagnostic MRI is used as a secondary imaging method after CT scans for pancreatic cancer detection and characterization. The superior soft-tissue contrast of MRI enables better tumor visualization and delineation.

### Imaging Details
- **Modality**: T1-weighted contrast-enhanced arterial phase MRI
- **Scanner**: Siemens scanners
- **Sequences**: Multiple sequences available including T1, T2, diffusion-weighted, and dynamic contrast-enhanced
- **Cases**: 92 annotated cases from clinical practice

### Data Characteristics
- **Total Dataset**: 489 diagnostic MRI scans (92 annotated + 367 unannotated + 5 development + 30 hidden test)
- **Data Source**: Radboud University Medical Center (Netherlands)
- **Format**: MHA files (.mha)
- **Annotations**: Binary tumor segmentation masks (0 = background, 1 = tumor)
- **License**: CC BY-NC-SA 4.0

## Dataset Split (Med-MLE-Bench)

For this benchmark, we use the 92 annotated cases:
- **Training Set**: 73 cases (80%)
- **Test Set**: 19 cases (20%)

Split is deterministic with `random_state=42` to ensure reproducibility across users.

## Goals

1. **Automate Tumor Delineation**: Reduce manual segmentation workload for clinicians
2. **Improve Diagnostic Accuracy**: Enable more precise tumor detection and characterization
3. **Support Treatment Planning**: Provide accurate tumor segmentation for surgical and radiotherapy planning
4. **Foster Innovation**: Support research in medical image segmentation, transfer learning, and few-shot learning

## Evaluation Metrics

### Primary Metric
- **Dice Similarity Coefficient (DSC)**: Measures overlap between predicted and ground truth segmentation

### Secondary Metrics
- **5mm Surface Dice (MSD)**: Surface-based metric with 5mm tolerance
- **Hausdorff Distance 95% (HD95)**: 95th percentile of maximum distances between surfaces
- **Mean Average Surface Distance (MASD)**: Average distance between predicted and ground truth surfaces
- **RMSE on Tumor Burden**: Root mean square error on tumor volume estimation

### Ranking Method
- Algorithms evaluated on each test case
- Metrics computed per case and averaged
- Overall ranking determined by mean position across all metrics
- Primary score is DSC for leaderboard ranking

## Submission Format

Your submission should include:
1. **CSV file** (`submission.csv`) with columns:
   - `patient_id`: Patient identifier (e.g., "10000_0001")
   - `prediction_file`: Filename of segmentation mask (e.g., "10000_0001.mha")

2. **Segmentation masks directory** containing:
   - Binary segmentation masks in MHA format
   - Same spatial dimensions as input images
   - Values: 0 = background, 1 = tumor
   - Filenames matching the `prediction_file` column in CSV

Example submission structure:
```
submission/
├── submission.csv
└── masks/
    ├── 10000_0001.mha
    ├── 10001_0001.mha
    └── ...
```

## Leaderboard

Top teams on the PANTHER Grand Challenge Task 1 leaderboard:

| Rank | Team | DSC | Mean Position |
|------|------|-----|---------------|
| 1st | ofdurugo | 0.7265 | 1.0 |
| 2nd | MoriiHuang | 0.6863 | 4.8 |
| 3rd | Kyriaki_Kolpetinou | 0.6744 | 5.0 |

## Challenge Information

- **Year**: 2024
- **Organizers**: Radboud University Medical Center & Odense University Hospital
- **Prize**: €500 for 1st place
- **Publication**: Top performers will have publication opportunities
- **Data Repository**: Zenodo (https://zenodo.org/records/15192302)
- **Grand Challenge Link**: https://panther.grand-challenge.org/

## References

For more details about the PANTHER challenge and MRI imaging for pancreatic cancer, visit the Grand Challenge page.
