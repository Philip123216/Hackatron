# Hackatron Project

This repository contains a machine learning pipeline for image classification using CNN models.

## Project Structure

```
Hackatron/
├── data/                  # Data directory
│   ├── raw/               # Raw downloaded images
│   ├── annotated/         # Annotated images (yes/no/skip)
│   └── processed/         # Processed data ready for training
├── models/                # Saved model files
├── src/                   # Source code
│   ├── data/              # Data acquisition and processing scripts
│   ├── annotation/        # Image annotation tools
│   ├── training/          # Model training scripts
│   ├── evaluation/        # Model evaluation scripts
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for exploration and visualization
├── tests/                 # Test scripts
└── README.md              # Project documentation
```

## Usage

### Data Acquisition

Scripts for downloading images from the Annotator service.

### Data Annotation

Tools for manually annotating downloaded images.

### Model Training

Scripts for training CNN models on annotated data.

### Model Evaluation

Scripts for evaluating trained models.

## Requirements

See requirements.txt for a list of dependencies.