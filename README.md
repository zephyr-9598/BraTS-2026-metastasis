# Reliability-Weighted Supervision for Brain Metastasis Segmentation

<div align="center">
  <h3>Epistemic Uncertainty-Guided Supervision Weighting for Brain Metastasis Sub-region Segmentation</h3>
  
  [![BraTS](https://img.shields.io/badge/BraTS-2026-0066cc.svg)](https://www.synapse.org/#!Synapse:syn53708249/wiki/627703)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
</div>

## 📖 Overview

This repository contains the official implementation of **URW-Met** (Uncertainty Reliability Weighting for Metastasis), a framework for brain metastasis segmentation that modulates per-class supervision based on epistemic uncertainty estimated from cross-validation fold ensemble disagreement.

### Key Contributions

- **Reliability-Weighted Supervision**: Down-weights training signal for classes with high inter-fold disagreement, reducing the influence of ambiguous subregions on model optimization.
- **No Architectural Changes**: Built on top of nnU-Net v2 without any modifications to the network architecture, preprocessing, or inference pipeline.
- **Task-Specific Reliability**: Demonstrates that pure epistemic weighting outperforms aleatoric uncertainty estimation and frequency correction for brain metastasis segmentation.
- **State-of-the-Art Results**: Achieves significant improvements on the BraTS 2026 Metastases validation set, with RC showing +11.5% improvement in LW Dice.

### 🎯 Performance Highlights

| Method | LW Dice (RC) | LW Dice (ET) | LW Dice (All Lesions) |
|--------|-------------|--------------|----------------------|
| nnU-Net Baseline | 0.5225 | 0.6947 | 0.6911 |
| **URW-Met (Ours)** | **0.5825** | **0.7247** | **0.7021** |
| **Improvement** | **+11.5%** | **+4.3%** | **+1.6%** |

## 📋 Table of Contents

- [Overview](#-overview)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🧠 Methodology

### Problem Statement

Brain metastasis segmentation presents challenges in accurately delineating tumor subregions with high morphological variability and heterogeneous MRI appearance, particularly for:
- **Non-Enhancing Tumor Core (NETC)**: Poorly defined boundaries with surrounding SNFH
- **Resection Cavity (RC)**: Highly variable post-surgical structure absent in many patients

### Our Approach

#### 1. Epistemic Uncertainty Estimation

Using nnU-Net's 5-fold cross-validation, we train five independent models on non-overlapping 80/20 data splits. The per-class epistemic uncertainty map is computed as:

$$u_i^c = \text{Var}_k\left[p_i^{(k,c)}\right]$$

where $p_i^(k,c)$ represents the softmax probability for class $c$ from fold $k$.

#### 2. Reliability Weight Computation

For each case `i` and class `c`, we:
- Compute inter-fold variance normalized across classes
- Assign minimum weight `ε = 0.05` to absent classes
- Apply per-case normalization to ensure consistent scaling

#### 3. Weighted Loss Function

The weighted loss combines reliability-weighted cross-entropy and Dice loss:

$$L_rel = 1/|P| * Σ(w_i^c * L_CE) + 1/C * Σ(1 - (2*Σ(w_i^c * p^c * y^c) + ε)/(Σ(w_i^c * (p^c + y^c)) + ε))$$

The **cross-entropy term** drives the majority of the improvement (86.39%), while the **Dice term** provides complementary regularization (12.58%).

### 🧪 Key Findings

| Configuration | Mean Dice | Δ from Baseline |
|---------------|-----------|-----------------|
| nnU-Net (E1) | 0.5569 | — |
| **Pure Epistemic (E2a)** | **0.6001** | **+0.0432** |
| Epistemic + Frequency (E2b) | 0.5439 | -0.0130 |
| Pure Aleatoric (E4a) | 0.5298 | -0.0271 |
| Aleatoric + Frequency (E4b) | 0.5449 | -0.0120 |

**Key Insight**: Pure epistemic weighting outperforms both aleatoric uncertainty estimation and frequency correction, suggesting that in the metastasis setting, difficulty stems from appearance variability rather than representation deficit.

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- NVIDIA GPU with 16GB+ VRAM (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/zephyr-9598/BraTS-2026-metastasis.git
cd BraTS-2026-metastasis

# Create conda environment
conda create -n brats_met python=3.10
conda activate brats_met

# Install dependencies
pip install -r requirements.txt

# Install nnU-Net v2
pip install nnunetv2
```

### Dependencies

```
torch>=1.10.0
nnunetv2>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
SimpleITK>=2.2.0
batchgenerators>=0.25
```

## 📊 Usage

### Data Preparation

1. Organize your BraTS 2026 dataset in the following structure:

```
dataset/
├── imagesTr/
│   ├── case_0000.nii.gz  # T1
│   ├── case_0001.nii.gz  # T1ce
│   ├── case_0002.nii.gz  # T2
│   └── case_0003.nii.gz  # FLAIR
├── labelsTr/
│   └── case.nii.gz        # Segmentation labels
└── dataset.json           # Dataset configuration
```

2. Preprocess the dataset using nnU-Net:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID -tr nnUNEtReliabilityweights
```

### Training

#### Step 1: Train Baseline 5-Fold Models

```bash
# Train all 5 folds
for fold in 0 1 2 3 4; do
  python train.py --fold $fold --output_dir ./results/baseline
done
```

#### Step 2: Train with Reliability-Weighted Supervision

```bash
# Train with pure epistemic weighting (E2a)
python train.py \
  --fold 0 \
  --weight_type epistemic \
  --weight_dir ./weights \
  --output_dir ./results/epistemic

# Train with aleatoric weighting (E4a)
python train.py \
  --fold 0 \
  --weight_type aleatoric \
  --weight_dir ./weights \
  --output_dir ./results/aleatoric
```

### Inference

```bash
python inference.py \
  --model_dir ./results/epistemic \
  --input_dir ./dataset/imagesTs \
  --output_dir ./predictions
```


## 📈 Results

### Official BraTS 2026 Validation Results

| Label | LW Dice | LW NSD@1.0 | F1 (Small) | F1 (All) |
|-------|---------|------------|------------|----------|
| **ET (3)** | **0.7247** | **0.7857** | 0.4598 | 0.8134 |
| **RC (4)** | **0.5825** | **0.4874** | 0.0825 | 0.5366 |
| **Core (1+2+3)** | **0.7454** | **0.7810** | 0.5024 | 0.8335 |
| **All Lesions (1+2+3+4)** | **0.7021** | **0.7348** | 0.4115 | 0.8236 |

### Comparison with Baseline

<div align="center">

| Metric | ET | RC | Core | All Lesions |
|--------|----|----|------|-------------|
| LW Dice Δ | **+4.3%** | **+11.5%** | **+3.0%** | **+1.6%** |
| LW NSD Δ | **+4.0%** | **+12.5%** | **+1.0%** | **+2.8%** |

</div>

## 📁 Repository Structure

```
BraTS-2026-metastasis/
├── src/
│   ├── custom_trainer.py         # Custom nnUNetTrainer with reliability weighting
│   ├── uncertainty.py            # Epistemic and aleatoric uncertainty estimation
│   ├── loss.py                   # Reliability-weighted loss functions
│   ├── weights.py                # Reliability weight computation and normalization
│   └── utils.py                  # Utility functions
├── scripts/
│   ├── compute_weights.py        # Compute reliability weights from fold models
│   ├── train.py                  # Training script
│   ├── inference.py              # Inference script
│   └── evaluate.py               # Evaluation script
├── assets/
│   ├── qualitative_comparison.png
│   └── architecture.png
├── requirements.txt
└── README.md
```

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{paravila2026epistemic,
  title={Epistemic Uncertainty-Guided Supervision Weighting for Brain Metastasis Sub-region Segmentation},
  author={Paravila, Ajesh Saviour},
  journal={BraTS 2026 Challenge},
  year={2026},
  url={https://github.com/zephyr-9598/BraTS-2026-metastasis}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **National Health Research Institutes (NHRI), Taiwan**: For providing computational resources and support
- **Dr. Maxim Solovchuk**: For guidance and supervision
- **BraTS 2026 Organizers**: For organizing the challenge and providing the dataset
- **nnU-Net Team**: For their excellent framework that served as the foundation for this work

## 🔗 Related Links

- [BraTS 2026 Challenge](https://www.synapse.org/#!Synapse:syn53708249/wiki/627703)
- [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnUNet)
- [BraTS 2026 Paper](https://doi.org/10.5281/ZENODO.19714728)

## 📧 Contact

**Ajesh Saviour Paravila**
- Email: ajesh.saviour@gmail.com
- GitHub: [@zephyr-9598](https://github.com/zephyr-9598)
- Institute: Institute of Biomedical Engineering and Nanomedicine, National Health Research Institutes, Taiwan

---

<div align="center">
  Made with ❤️ by Ajesh Saviour Paravila
</div>
