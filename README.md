# CV Dataset Toolkit

A Python + OpenCV pipeline for **automated image data augmentation** and **quality validation** — designed for preparing and quality-checking datasets for AI/Computer Vision model training.

## Features

### 🔄 Data Augmentation
- **Geometric transforms**: Rotation, horizontal/vertical flip, random cropping
- **Color transforms**: Brightness adjustment, contrast adjustment, saturation shift
- **Noise injection**: Gaussian noise, salt-and-pepper noise
- **Blur simulation**: Gaussian blur, motion blur
- **Combined augmentation**: Apply random combinations for diverse training data

### ✅ Quality Validation
- **Blur detection**: Laplacian variance-based sharpness scoring
- **Brightness analysis**: Mean luminance checks (too dark / too bright)
- **Resolution check**: Minimum dimension enforcement
- **Contrast analysis**: Standard deviation-based contrast scoring
- **Aspect ratio validation**: Flags extreme aspect ratios
- **Comprehensive report**: JSON/CSV quality reports with pass/fail per image

### 📦 Structured Export
- Augmented images saved with descriptive filenames
- JSON metadata logs with augmentation parameters for reproducibility
- CSV quality reports for dataset-level analysis
- Configurable output directory structure

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Augment Images
```bash
# Augment all images in a directory (5 variants per image)
python main.py augment --input ./sample_images --output ./output/augmented --count 5

# Augment with specific transforms only
python main.py augment --input ./sample_images --output ./output/augmented --transforms rotate flip brightness noise
```

### Validate Image Quality
```bash
# Validate all images in a directory
python main.py validate --input ./sample_images --output ./output/reports

# Validate with custom thresholds
python main.py validate --input ./sample_images --min-resolution 640 --blur-threshold 80
```

### Full Pipeline (Augment + Validate)
```bash
# Run augmentation then validate the augmented output
python main.py pipeline --input ./sample_images --output ./output --count 5
```

## Output Structure

```
output/
├── augmented/
│   ├── image1_rot90.jpg
│   ├── image1_flip_h.jpg
│   ├── image1_bright_1.3.jpg
│   └── ...
├── reports/
│   ├── quality_report.json
│   └── quality_report.csv
└── metadata/
    └── augmentation_log.json
```

## Tech Stack

- **Python 3.8+**
- **OpenCV** — Image processing and computer vision operations
- **NumPy** — Numerical operations for pixel manipulation
- **Standard library** — argparse, json, csv, pathlib

## Author

Vikas Kumar — [GitHub](https://github.com/Vikaumar)
