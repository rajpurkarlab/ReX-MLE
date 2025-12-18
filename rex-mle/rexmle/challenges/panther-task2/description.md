# PANTHER Task 2: Pancreatic Tumor Segmentation for MR-Linac Adaptive Radiotherapy

**Grand Challenge:** https://panther.grand-challenge.org/

## Overview

PANTHER Task 2 focuses on pancreatic tumor segmentation from T2-weighted MRI acquired on MR-Linac systems during radiotherapy. This task addresses the critical need for real-time tumor delineation in adaptive radiotherapy workflows.

This is the **first publicly available dataset** for pancreatic MR-Linac imaging, enabling AI development for online adaptive radiotherapy planning.

## Task Description

### Clinical Context
MR-Linac (MRI-guided Linear Accelerator) combines real-time MRI imaging with radiation delivery, enabling adaptive radiotherapy. During each treatment session, clinicians must quickly segment the tumor to adapt the radiation plan to daily anatomical changes. **Inference time is critical** for this task as it directly impacts treatment workflow efficiency.

### MR-Linac Treatment Workflow
1. Initial simulation and treatment planning
2. Daily MRI setup to verify current anatomy
3. **Online tumor segmentation** (target of this task)
4. Treatment plan adaptation based on current anatomy
5. Radiation delivery with continuous tumor monitoring
6. Optional post-treatment imaging

### Imaging Details
- **Modality**: T2-weighted MRI
- **Scanner**: Elekta Unity MR-Linac system
- **Sequences**: Limited sequences compared to diagnostic MRI (T2-weighted only)
- **Cases**: 50 annotated cases from radiotherapy sessions

### Data Characteristics
- **Total Dataset**: 84 MR-Linac scans (50 annotated + 4 development + 30 hidden test)
- **Data Source**: Odense University Hospital (Denmark)
- **Format**: MHA files (.mha)
- **Annotations**: Binary tumor segmentation masks (0 = background, 1 = tumor)
- **License**: CC BY-NC-SA 4.0

## Dataset Split (Med-MLE-Bench)

For this benchmark, we use the 50 annotated cases:
- **Training Set**: 40 cases (80%)
- **Test Set**: 10 cases (20%)

Split is deterministic with `random_state=42` to ensure reproducibility across users.

## Key Challenges

1. **Limited Image Quality**: MR-Linac images have lower resolution and contrast compared to diagnostic MRI
2. **Speed Requirements**: Segmentation must be fast enough for clinical workflow (typically <5 minutes)
3. **Domain Shift**: Different imaging characteristics compared to diagnostic MRI (Task 1)
4. **Daily Anatomical Changes**: Tumor position and shape vary between treatment sessions

## Goals

1. **Enable Adaptive Radiotherapy**: Provide fast, accurate tumor segmentation for online plan adaptation
2. **Reduce Clinical Workload**: Automate manual contouring during treatment sessions
3. **Improve Treatment Precision**: Enable precise radiation delivery based on daily anatomy
4. **Support Transfer Learning Research**: Bridge the gap between diagnostic and treatment imaging domains

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

**Note**: While not explicitly measured in the leaderboard, **inference time is a critical practical consideration** for clinical deployment.

## Submission Format

Your submission should include:
1. **CSV file** (`submission.csv`) with columns:
   - `patient_id`: Patient identifier (e.g., "10303")
   - `prediction_file`: Filename of segmentation mask (e.g., "10303.mha")

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
    ├── 10303.mha
    ├── 10304.mha
    └── ...
```

## Leaderboard

Top teams on the PANTHER Grand Challenge Task 2 leaderboard:

| Rank | Team | DSC | Mean Position |
|------|------|-----|---------------|
| 1st | MIC-DKFZ | 0.5289 | 1.6 |
| 2nd | BreizhSeg | 0.4910 | 2.4 |
| 2nd | LiboZhang | 0.4814 | 2.4 |

Note: Task 2 achieves lower DSC scores than Task 1 due to the challenging nature of MR-Linac imaging (lower contrast, limited sequences).

## Challenge Information

- **Year**: 2024
- **Organizers**: Radboud University Medical Center & Odense University Hospital
- **Prize**: €500 for 1st place
- **Publication**: Top performers will have publication opportunities
- **Data Repository**: Zenodo (https://zenodo.org/records/15192302)
- **Grand Challenge Link**: https://panther.grand-challenge.org/

## References

For more details about the PANTHER challenge, MR-Linac technology, and adaptive radiotherapy, visit the Grand Challenge page.
