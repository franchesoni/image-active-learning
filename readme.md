# Binary Annotation of Images with Active Learning

A fast, efficient tool for annotating images with binary labels using active learning to maximize annotation efficiency.

## Overview

This tool provides a web-based interface for binary image annotation with active learning strategies that prioritize the most informative samples, helping you:

- Build high-quality datasets quickly
- Focus human effort on the most valuable samples
- Monitor model improvement in real time

## Use Cases

- **Error identification**: Annotate correct/incorrect images to train error-checking models
- **Detection refinement**: Filter bounding box proposals from general detection models (like SAM)  
- **Object classification**: Quickly build binary classifiers (e.g., defective vs. non-defective parts)
- **Data cleaning**: Identify and remove outliers or mislabeled images

## Active Learning Strategies

The tool offers two methods for selecting samples:

1. **Entropy maximization (certainty)**: Prioritizes samples where the model is most uncertain (probability near 0.5)
2. **Maximum probability**: Optimized for finding rare positives/negatives by focusing on samples with high model confidence

## Features

- **Keyboard shortcuts** for rapid annotation
- **Real-time error metrics** to track model improvement
- **Progress tracking** with visual indicators
- **Annotation reverting** capability
- **Prefetching** for smoother experience
- **Optimized sample selection** using active learning

## Requirements

- Python 3.7+
- Required packages: `pip install -r requirements.txt`
- A list of images paths OR image paths with bounding boxes (see `instance_references_EXAMPLE.txt`)

## Quick Start

### 1. Compute Features

Feature extraction is critical for active learning performance. Choose one of these methods:

```bash
# Using DINOv2 (recommended)
python compute_features.py --refs='instance_references_EXAMPLE.txt'

# Specify batch size and workers based on your hardware
python compute_features.py --refs='your_paths_file.txt' --batch_size=32 --num_workers=4

# Generate features with a custom output path
python compute_features.py --refs='your_paths_file.txt' --out='custom_features.npy'
```
### 2. Launch Annotation Interface
```bash
uvicorn app:app --port 8000
```
Then open your browser to http://127.0.0.1:8000 to start annotating.

### 3. Annotation Controls
- A / Left Arrow: Label as Negative
- D / Right Arrow: Label as Positive
- Backspace: Revert last annotation
- H / ?: Show keyboard shortcuts

## Performance Tips
- Start with just 10-20 minutes of annotation to get early results
- Take short breaks to avoid annotation fatigue
- The model improves as you annotate more samples, so efficiency increases over time
- For highly imbalanced data, use "maximum probability" strategy to find rare classes

## Advanced Configuration
Edit the constants at the top of app.py to customize paths and behavior:

```python
REFS_FILE = "your_custom_paths.txt"
FEATURES_FILE = "your_features.npy"
ANNOTATIONS_FILE = "your_annotations.csv"
```

## Interpreting the Error Chart
The error visualization shows:

- Red points: Individual Brier scores (lower is better)
- Red line: Average Brier score
- Blue line: Error rate (incorrect predictions)
As annotation progresses, both lines should trend downward, indicating model improvement.

