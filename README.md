# PolyScan: AI-Powered Polyp Segmentation

![Medical AI](https://img.shields.io/badge/Application-Medical_AI-blue)
![License](https://img.shields.io/badge/License-MIT-green)

PolyScan is a deep learning system for automatic polyp segmentation in colonoscopy images, implementing state-of-the-art architectures with a focus on clinical applicability.

## Features

- **Medical-Grade Segmentation**
  - FCN (Fully Convolutional Network) and UNet architectures
  - Comprehensive clinical metrics (Dice, IoU, Sensitivity, Specificity)
  - Boundary accuracy measurement (Hausdorff Distance 95)

- **Reproducible Workflow**
  - YAML configuration-driven experiments
  - Automatic model versioning
  - MLflow integration for experiment tracking

- **Clinical Visualization**
  - Overlay visualization with adjustable transparency
  - Multi-panel result comparisons
  - DICOM-compatible output formats

## Installation

**Clone Repository**
```bash
git clone --recurse-submodules https://github.com/orkhanshirin/PolyScan.git
cd PolyScan
```
Optional (if you want to access the data used for training)
```bash
git lfs pull
```
**Install Dependencies**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
### Training

```bash
python3 src/training/train.py --config configs/train.yml
```
Sample `configs/train.yml`
```yml
model:
  name: "unet"
  params:
    in_channels: 3
    init_filters: 64
training:
  epochs: 100
  batch_size: 8
data:
  metadata_file: "metadata.csv"
```

### Evaluation

```bash
python3 src/evaluation/evaluate.py --config configs/eval.yml
```

### Inference

```bash
python3 src/inference.py \
  --config configs/inference.yml \
  --image data/samples/polyp_case_15.png \
  --output results/prediction.png
```

## Configuration

Key configuration files:

    configs/unet.yml: Training hyperparameters for UNet
    configs/fcn.yml: Training hyperparameters for FCN
    configs/eval.yml: Evaluation metrics setup
    configs/inference.yml: Clinical deployment settings

Example configuration structure:
```yml
# model Configuration
model:
  name: "unet"
  experiment_name: "unet"
  use_latest: True

# data parameters
data:
  input_size: [256, 256]
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

# visualization
visualization:
  overlay_color: [255, 0, 0]  # BGR format
  alpha: 0.4
```

## Project Structure

```bash
|PolyScan
├── configs
│   ├── eval.yml
│   ├── fcn.yml
│   ├── inference.yml
│   └── unet.yml
├── data_tools
│   ├── __init__.py
│   ├── dataset.py
│   ├── loader.py
│   ├── split.py
│   └── validate.py
├── src
│   ├── architectures
│   │   ├── __init__.py
│   │   ├── fcn.py
│   │   └── unet.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   └── evaluate.py
│   ├── metrics
│   │   ├── __init__.py
│   │   ├── clinical.py
│   │   ├── dice.py
│   │   ├── hd95.py
│   │   └── iou.py
│   ├── training
│   │   ├── config
│   │   │   ├── __init__.py
│   │   │   ├── defaults.py
│   │   │   └── parser.py
│   │   ├── core
│   │   │   ├── __init__.py
│   │   │   ├── callbacks.py
│   │   │   └── trainer.py
│   │   ├── tracking
│   │   │   ├── __init__.py
│   │   │   └── mlflow.py
│   │   ├── __init__.py
│   │   └── train.py
│   ├── visualization
│   │   ├── __init__.py
│   │   ├── overlay.py
│   │   └── plotting.py
│   ├── __init__.py
│   ├── helpers.py
│   └── inference.py
├── Dockerfile
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```


## Clinical Validation Metrics

| **Metric**      | **Description**                  | **Target Value** |
|----------------|----------------------------------|-----------------|
| **Dice Score** | Region overlap accuracy         | **> 0.85** |
| **IoU**        | Strict spatial agreement        | **> 0.75** |
| **Sensitivity**| Polyp detection rate           | **> 0.90** |
| **HD95**       | Boundary segmentation accuracy | **< 5.0mm** |

## Visualization example
![segmentation](example/output_5.png)

### Color Convention:
* Red: Model predictions
* Green: Ground truth annotations
* Blue: Uncertain regions (when using uncertainty maps)


## Contributing

We welcome medical and technical contributions:
* Report clinical validation results via GitHub Issues
* Submit PRs for new architectures in src/architectures/
* Improve documentation for clinical deployment scenarios

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
* Medical image preprocessing adapted from [MONAI](https://monai.io/) frameworks
* Initial architecture designs inspired by [MICCAI](https://miccai.org/) publications
* Dataset handling patterns from [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) dataset guidelines
