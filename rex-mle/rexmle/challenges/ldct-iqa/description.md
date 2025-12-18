# Low-dose Computed Tomography Perceptual Image Quality Assessment

**Grand Challenge:** https://ldctiqac2023.grand-challenge.org/ldctiqac2023/

## Overview

The LDCT-IQA Challenge focuses on developing no-reference image quality assessment (NR-IQA) models that correlate well with radiologists' perceptual scores on low-dose CT images. This challenge addresses a critical gap in medical imaging: the lack of reliable, reference-free quality metrics for CT images containing complex artifacts and noise.

Traditional metrics like PSNR and SSIM have insufficient correlation with radiologists' opinions and require pristine reference images that are often impossible to obtain in clinical settings due to radiation dose concerns. This challenge aims to establish a standard IQA metric for CT imaging that works without reference images and correlates strongly with radiologist perception.

## Task Description

### Clinical Context
Low-dose CT imaging reduces radiation exposure to patients but introduces image quality degradation through various artifacts. Assessing image quality is crucial for optimizing radiation dose and developing novel restoration algorithms. However, obtaining high-quality reference images for comparison is impractical due to radiation safety concerns.

### Imaging Details
- **Modality**: Low-dose abdominal CT
- **Window Settings**: Soft-tissue windows (width/level: 350/40)
- **Artifacts**: Combination of sparse view streak artifacts and noise
- **Generation Method**:
  - Reduced projections per rotation
  - Reduced X-ray current
- **Radiologist Evaluation**: 5 experienced radiologists scored each image
- **Ground Truth**: Final perceptual scores averaged across 5 radiologists

### Data Characteristics
- **Data Source**: Original dataset from NIH Low Dose CT Grand Challenge (Mayo Clinic)
- **Format**: CT images with quality degradation
- **Annotations**: Human perceptual quality scores from radiologists using diagnostic criteria
- **License**: CC BY-NC-SA 4.0

## Dataset Split (Med-MLE-Bench)

The dataset is split into:
- **Training Set**: Images with radiologist quality scores for model development
- **Test Set**: Held-out images for evaluation

Split is deterministic with `random_state=42` to ensure reproducibility across users.

## Goals

1. **Develop NR-IQA Models**: Create no-reference quality metrics for CT images with complex artifacts
2. **Correlate with Clinical Perception**: Align automated scores with radiologists' diagnostic quality assessment
3. **Enable Clinical Application**: Provide metrics usable in real-world scenarios without reference images
4. **Standardize CT IQA**: Establish benchmark performance for CT image quality assessment

## Evaluation Metrics

### Primary Metrics
The final overall score is calculated by summing correlation coefficients:

```python
from scipy.stats import pearsonr, spearmanr, kendalltau

aggregate_results = dict()
aggregate_results["plcc"] = abs(pearsonr(total_pred, total_gt)[0])
aggregate_results["srocc"] = abs(spearmanr(total_pred, total_gt)[0])
aggregate_results["krocc"] = abs(kendalltau(total_pred, total_gt)[0])
aggregate_results["overall"] = abs(pearsonr(total_pred, total_gt)[0]) + abs(spearmanr(total_pred, total_gt)[0]) +  abs(kendalltau(total_pred, total_gt)[0])
```

- **PLCC (Pearson Linear Correlation Coefficient)**: Measures linear correlation between predicted and radiologist scores
- **SROCC (Spearman Rank Order Correlation Coefficient)**: Measures monotonic relationship (rank-based)
- **KROCC (Kendall Rank Order Correlation Coefficient)**: Measures ordinal association between rankings

### Overall Score
**Overall Score = PLCC + SROCC + KROCC**

Higher values indicate better correlation with radiologist perception. All coefficients are calculated using absolute values.

## Submission Format

Your submission should be a **CSV file** (`submission.csv`) with columns:
- `image_id`: Image identifier
- `quality_score`: Predicted perceptual quality score

Example:
```csv
image_id,quality_score
img_001,3.45
img_002,2.87
img_003,4.12
...
```

## Challenge Information

- **Year**: 2023
- **Conference**: MICCAI 2023
- **Organizers**:
  - Wonkyeong Lee (Ewha Womans University, South Korea)
  - Fabian Wagner (FAU Erlangen-Nürnberg, Germany)
  - Prof. Andreas Maier (FAU Erlangen-Nürnberg, Germany)
  - Prof. Adam Wang (Stanford University, USA)
  - Prof. Jongduk Baek (Yonsei University, South Korea)
  - Prof. Scott S. Hsieh (Mayo Clinic, USA)
  - Prof. Jang-Hwan Choi (Ewha Womans University, South Korea)
- **Data Repository**: Zenodo (https://zenodo.org/record/7833096)
- **Grand Challenge Link**: https://ldctiqac2023.grand-challenge.org/ldctiqac2023/

## Important Dates

- Training data release: 20/04/2023
- Registration deadline: 14/07/2023
- Test phase opens: 14/07/2023
- Submission deadline: 28/07/2023 11:59 p.m. (UTC)
- Announcement of winners: 21/08/2023

## Funding

This research was supported by:
- Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2022-00155966)
- National Research Foundation of Korea (NRF-2022R1A2C1092072)
- Korea Medical Device Development Fund (Project Number: 1711174276, RS-2020-KD000016)

## References

**Original Low-dose CT Dataset:**
NIH grants EB017095 and EB017185 (Cynthia McCollough, PI) from the National Institute of Biomedical Imaging and Bioengineering, and American Association of Physicists in Medicine, Low Dose CT Grand Challenge Dataset.

**Citation:**
Wonkyeong Lee, Fabian Wagner, Andreas Maier, Adam Wang, Jongduk Baek, Scott S. Hsieh, & Jang-Hwan Choi. (2023). Low-dose Computed Tomography Perceptual Image Quality Assessment Grand Challenge Dataset (MICCAI 2023) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7833096
