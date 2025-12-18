# [Challenge Name]

## Overview

[Brief description of the challenge, what it aims to solve, and why it's important]

## Dataset

### Images

- **Modality**: [CT/MRI/X-ray/etc.]
- **Body Part**: [Chest/Brain/Abdomen/etc.]
- **Format**: [DICOM/NIfTI/MHA/PNG]
- **Dimensions**: [Typical image size]
- **Training Set**: [Number] images
- **Test Set**: [Number] images

### Labels

[Description of what is being predicted]

- **Task Type**: [Classification/Segmentation/Detection/Regression]
- **Classes**: [List of classes or description of labels]

### Data Structure

```
public/
├── train/
│   ├── images/
│   │   ├── image_001.dcm
│   │   └── ...
│   └── train.csv  # Labels
├── test/
│   ├── images/
│   │   ├── image_100.dcm
│   │   └── ...
│   └── test.csv  # IDs only (no labels)
└── sample_submission.csv
```

## Evaluation

### Primary Metric

[Name of metric, e.g., AUC, Dice coefficient, mAP]

[Explanation of how the metric is calculated]

### Submission Format

Submissions should be a CSV file with the following columns:

```csv
image_id,prediction
image_001,0.95
image_002,0.12
...
```

- `image_id`: Image identifier (must match test set)
- `prediction`: [Description of what prediction represents]

For classification: prediction should be probability between 0 and 1
For segmentation: submit segmentation masks as separate files
For detection: [Describe bounding box format]

## Rules

1. You may use any publicly available data for pre-training
2. You may NOT use additional labeled data from this specific task
3. Submissions are limited to [number] per day

## Timeline

- **Start Date**: [Date]
- **End Date**: [Date]
- **Prizes**: [Prize information if applicable]

## Citation

If you use this dataset, please cite:

```
[Citation information]
```

## Data Source

- **Grand Challenge**: [URL]
- **Zenodo**: [DOI or URL]
- **Original Paper**: [Citation]

## Additional Resources

- [Link to related papers]
- [Link to baseline code]
- [Link to discussion forum]
