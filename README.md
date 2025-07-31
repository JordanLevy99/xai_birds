# XAI Birds: Multi-Task Learning for Bird Attribute Classification

A deep learning project that uses multi-task learning to classify bird attributes from images, with explainable AI (XAI) capabilities using Captum for model interpretability.

## Overview

This project implements a multi-task learning approach to classify various bird attributes including:
- Bill shape
- Wing color
- Breast pattern
- Size
- Species
- Family

The model uses a VGG16 backbone with custom classification heads for each attribute, enabling simultaneous learning of multiple bird characteristics from a single image.

## Features

- **Multi-Task Learning**: Train multiple classification tasks simultaneously using shared representations
- **Explainable AI**: Generate Grad-CAM visualizations to understand model decisions
- **Flexible Configuration**: JSON-based configuration for easy experimentation
- **Comprehensive Evaluation**: Training/validation loss tracking and visualization

## Project Structure

```
├── config/                 # Training configurations
│   └── train.json         # Main training configuration
├── figures/               # Generated training plots and visualizations
├── logs/                  # Training logs
├── notebooks/             # Jupyter notebooks for experimentation
│   ├── Captum Demo.ipynb              # XAI demonstrations
│   ├── Multi-Task Eval.ipynb          # Model evaluation
│   └── multi-task learning.ipynb     # Training experiments
├── processed_data/        # Preprocessed datasets
├── resources/             # Research papers and references
├── src/                   # Source code
│   ├── models/
│   │   └── multi_task_model.py       # Multi-task model architecture
│   ├── bird_dataset.py               # Dataset handling
│   ├── XAI_BirdAttribute_dataloader.py  # Data loading utilities
│   ├── multi_task_training.py        # Training pipeline
│   └── utils.py                       # Helper functions
└── run.py                 # Main training script
```

## Installation

1. Clone this repository:
```bash
git clone <repository_url>
cd xai_birds
```

2. Install required dependencies:
```bash
pip install torch torchvision
pip install captum
pip install numpy pandas matplotlib scikit-image
pip install opencv-python pillow
```

## Usage

### Training

Run training with the default configuration:
```bash
python run.py train
```

The training configuration can be modified in `config/train.json`:
```json
{
    "preload": true,
    "preload_file": "images-families.pkl",
    "species_file": "classes-families.txt",
    "attrs": null,
    "species": true,
    "epochs": 190,
    "lr": 5e-5,
    "patience": 25,
    "batch_size": 1,
    "print_freq": 1000,
    "test": false
}
```

### Model Architecture

The `MultiTaskModel` class implements:
- VGG16 backbone for feature extraction
- Shared fully connected layers (1000 → 512 → 256 → 256)
- Task-specific classification heads for each attribute
- Dropout regularization (0.25)
- Xavier normal weight initialization

### Explainable AI

Use the provided notebooks to generate Grad-CAM visualizations:
- `notebooks/Captum Demo.ipynb`: Interactive XAI demonstrations
- Visualizations are saved to `figures/` directory

### Evaluation

Training progress is automatically logged and visualized:
- Loss curves are saved as PNG files in `figures/`
- Training logs are saved in `logs/`
- Model checkpoints are saved after training

## Key Components

- **MultiTaskModel**: Main neural network architecture with shared encoder and task-specific heads
- **MultiTaskLossWrapper**: Handles loss computation across multiple tasks
- **MultiTaskTraining**: Training pipeline with validation and early stopping
- **Bird_Attribute_Loader**: Custom dataset loader for bird attributes

## Research Context

This project is inspired by multi-task learning research and includes references to relevant papers in the `resources/` directory. The approach enables learning shared representations across related tasks while maintaining task-specific capabilities.

## Results

Training results and model performance visualizations are automatically generated and saved in the `figures/` directory. The project supports tracking multiple experiments with different attribute combinations.