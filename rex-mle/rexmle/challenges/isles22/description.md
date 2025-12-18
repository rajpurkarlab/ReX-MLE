# ISLES'22 - Ischemic Stroke Lesion Segmentation Challenge

**Challenge Website:** https://isles22.grand-challenge.org/
**Year:** 2022
**Conference:** MICCAI 2022
**Data License:** CC BY 4.0
**Dataset:** https://zenodo.org/record/7960856
**Challenge Document:** https://zenodo.org/record/6362388

## Overview

The Ischemic Stroke Lesion Segmentation Challenge (ISLES'22) focuses on **multimodal MRI infarct segmentation in acute and sub-acute stroke**. This challenge evaluates automated methods for stroke lesion segmentation using DWI, ADC, and FLAIR MR modalities.

## Background

Infarct segmentation in ischemic stroke is crucial at different disease stages:

1. **Acute stages**: Guide treatment decision-making
   - Whether to reperfuse or not
   - Type of treatment selection

2. **Sub-acute and chronic stages**: Evaluate disease outcome
   - Patient clinical follow-up
   - Define optimal therapeutic and rehabilitation strategies
   - Maximize critical windows for recovery

## Task Definition

**Goal:** Automatically generate stroke lesion segmentation masks from multimodal MRI images.

**Input Modalities:**
- **DWI** (Diffusion Weighted Imaging, b=1000)
- **ADC** (Apparent Diffusion Coefficient map)
- **FLAIR** (Fluid Attenuated Inversion Recovery)

**Output:** Binary segmentation mask delineating ischemic stroke lesions

## Key Features of ISLES'22

### 1. Broader Lesion Spectrum
- **Not only large infarcts** but also **multiple embolic and/or cortical infarcts**
- Typical patterns seen after mechanical recanalization
- Variable lesion size and burden

### 2. Temporal Coverage
- **Pre-intervention MRI**: Very early disease state
- **Post-intervention MRI**: Sub-acute infarct patterns not restricted to single vessel territory

### 3. Enhanced Dataset
- **~3x more data than ISLES'15** (previous similar challenge)
- **250 expert-annotated cases** for training
- **150 hidden cases** for testing

### 4. Multi-Center Data
- Retrospective data from three stroke centers
- Multiple scanner types and field strengths
- Real-world clinical variability

## Differences from Previous ISLES Editions

ISLES'22 differs from previous challenge editions by:

1. **Targeting complex infarct patterns**
   - Not just single large lesions
   - Multiple embolic infarcts
   - Cortical infarcts

2. **Dual temporal assessment**
   - Pre-interventional imaging
   - Post-interventional imaging

3. **Increased data volume and diversity**
   - Larger dataset (~3x ISLES'15)
   - Multi-center acquisition
   - Variable anatomical locations

4. **Clinical relevance**
   - Growing interest in acute embolic infarct patterns
   - Post-interventional sub-acute patterns
   - Wider disease spectrum representation

## Clinical Significance

From a clinical perspective, ISLES'22 addresses:

- **Acute embolic infarct patterns** (pre-intervention)
- **Post-interventional sub-acute patterns** (after mechanical recanalization)
- **Multiple infarcts** not restricted to a single vessel territory
- **Variable lesion burden** across different anatomical locations

## Technical Challenges

Participants will face:

- **Wider ischemic stroke disease spectrum**
- **Variable lesion size and burden**
- **More complex infarct patterns**
- **Variable anatomically located lesions**
- **Multi-center data variability**

## Dataset Structure

### Training Data (250 cases)
Each case includes:
- 3 MRI modalities (DWI, ADC, FLAIR)
- Expert-level stroke lesion annotations
- Clinical metadata (when available)

### File Format
- **Format:** NIfTI (Neuroimaging Informatics Technology Initiative)
- **Convention:** BIDS (Brain Imaging Data Structure)
- **Native space:** No prior registration applied
- **Preprocessing:** Skull-stripping for patient de-identification

### Scanner Details
Images acquired using:
- **3T Philips** (Achieva, Ingenia)
- **3T Siemens** Verio
- **1.5T Siemens MAGNETOM** (Avanto, Aera)

## Challenge Phases

### Phase 1: Preliminary Docker Submission (Sanity Check)
- Test docker functionality on GC servers
- Multiple submissions allowed
- Tests on 3 training cases:
  - sub-strokecase0001
  - sub-strokecase0002
  - sub-strokecase0003
- **Not used for ranking**

### Phase 2: ISLES'22 Test Phase (Real Evaluation)
- **Single submission** only
- Evaluation on **150 hidden test cases**
- Results used for official ranking

### Ongoing Evaluation
- Challenge remains open
- **One docker submission per month** allowed
- Continuous evaluation on test set
- Test set remains unreleased

## Submission Format

Participants must submit:
- **Docker container** with segmentation algorithm
- Container processes input MRI images
- Outputs binary segmentation masks

Expected output:
- Binary masks (0 = background, 1 = stroke lesion)
- Same dimensions as input images
- NIfTI format (.nii.gz)

## Evaluation Metrics

**Primary Metric:** Dice Similarity Coefficient (DSC)

**Secondary Metrics:**
- Hausdorff Distance (HD)
- Precision
- Recall
- Lesion-wise F1 score

## Example Case Structure

```
sub-strokecase####/
├── sub-strokecase####_dwi.nii.gz       # DWI image
├── sub-strokecase####_adc.nii.gz       # ADC map
├── sub-strokecase####_flair.nii.gz     # FLAIR image
└── sub-strokecase####_msk.nii.gz       # Ground truth mask (training only)
```

## Clinical Application

Accurate automated stroke lesion segmentation enables:

1. **Treatment planning**
   - Rapid lesion quantification
   - Treatment decision support
   - Intervention selection

2. **Outcome prediction**
   - Lesion burden assessment
   - Recovery potential evaluation
   - Rehabilitation planning

3. **Research applications**
   - Large-scale stroke studies
   - Treatment efficacy evaluation
   - Biomarker development

## Citation

If you use this dataset, please cite:

```
@dataset{isles22_2023,
  author       = {ISLES Challenge Organizers},
  title        = {ISLES 2022 Challenge Dataset},
  year         = {2023},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7960856},
  url          = {https://doi.org/10.5281/zenodo.7960856}
}
```

## Resources

- **Challenge Website:** https://isles22.grand-challenge.org/
- **Training Data:** https://zenodo.org/record/7960856
- **Challenge Document:** https://zenodo.org/record/6362388
- **Docker Template:** Available on challenge GitHub repositories

## Contact

For questions about the challenge, please refer to the official ISLES'22 Grand Challenge page.
