# NMC Code

## 📚 **Related Paper**

This code is an implementation for the following paper:

**A Comparative Study of Lesion-Centered and Severity-Based Approaches to Diabetic Retinopathy Classification: Improving Interpretability and Performance**

- **Journal**: Biomedicines (MDPI)
- **DOI**: [https://doi.org/10.3390/biomedicines13061446](https://doi.org/10.3390/biomedicines13061446)
- **Full Paper**: [https://www.mdpi.com/2227-9059/13/6/1446](https://www.mdpi.com/2227-9059/13/6/1446)

---

## Overview

This repository contains the refactored code for the NMC system. The code has been cleaned by removing unnecessary outputs and duplicate code, keeping only the core functionality.

## 📁 Directory Structure

```
nmc_clean/
├── core/           # Core NMC modules
├── configs/        # Configuration files
├── notebooks/      # Refactored Jupyter notebooks
├── tools/          # Evaluation tools used in NMCS
└── README.md       # This file
```

## 🔧 Core Modules

### Models
- **EfficientNetV2**: EfficientNetV2-based models
  - `EfficientNetV2MModel`: Single-label classification
  - `EfficientNetV2MModelMulti`: Multi-label classification
- **ResNet**: ResNet-based models
  - `ResNet50Model`: Single-label classification
  - `ResNet50MultiHeadModel`: Multi-label classification
- **FGMaxxVit**: FGMaxxVit-based models
  - `FGMaxxVit`: Single-label classification
  - `FGMaxxVit_Multi`: Multi-label classification
- **TestCNN**: Simple CNN model for testing

### Utils
- **augmentations.py**: Data augmentation functions
- **losses.py**: Loss functions
- **metrics.py**: Evaluation metrics
- **optimizers.py**: Optimization algorithms
- **schedulers.py**: Learning rate schedulers
- **utils/**: Other utility functions

## 📊 Configuration Files

- **NMC.yaml**: NMC dataset training configuration
- **APTOS.yaml**: APTOS dataset training configuration
- **ODIR.yaml**: ODIR dataset training configuration
- **Multi_Task.yaml**: Multi-task learning configuration

## 📓 Jupyter Notebooks (Core NMCS Files)

### NMC Related
- **NMC.ipynb**: Basic NMC model training and evaluation
- **NMC_singlelabel.ipynb**: Single-label NMC training
- **NMC_labelchain.ipynb**: Label chain-based NMC training
- **NMC_confusion.ipynb**: Confusion matrix analysis

### APTOS Related
- **APTOS.ipynb**: Basic APTOS model training
- **APTOS_singlelabel.ipynb**: Single-label APTOS training
- **APTOS_NMC_finetuning.ipynb**: APTOS fine-tuning with NMC
- **NMC_APTOS_finetuning.ipynb**: NMC fine-tuning with APTOS

### Visualization and Analysis
- **NMC_APTOS_visualization.ipynb**: Visualization tools
- **NMC_APTOS_gradcam.ipynb**: Grad-CAM analysis
- **NMC_APTOS_OSM.ipynb**: OSM (Object Saliency Map) analysis

### Special Models
- **NMC_APTOS_BIFPN.ipynb**: BIFPN (Bidirectional Feature Pyramid Network) model
- **NMC_APTOS_FPN.ipynb**: FPN (Feature Pyramid Network) model

## 🛠️ Tools (Actually Used in NMCS)

- **val.py**: Model evaluation tool (includes `evaluate_epi` function)
- **episodic_utils.py**: Episodic learning utilities

## 🚀 Quick Start

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install individually:
   ```bash
   pip install torch torchvision torchaudio
   pip install numpy pandas scipy scikit-learn
   pip install opencv-python Pillow matplotlib seaborn
   pip install tqdm PyYAML tabulate jupyter
   ```

2. **Configuration Check**
   - Check data paths and model settings in `configs/NMC.yaml`

3. **Run Notebooks**
   - Execute desired notebooks from the `notebooks/` folder

## ⚠️ Important Notes

- This code is a **refactored version** that includes only the core functionality actually used in NMCS
- **Original code remains unchanged**
- **Unnecessary files have been removed** (contrastive_proto, multi_task, etc.)
- Set correct paths in configuration files before execution
- GPU environment is required (CUDA support)

## 🔗 Dependencies

- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy
- PIL (Pillow)
- OpenCV
- tabulate
- tqdm
- PyYAML

## 📝 License

Follows the license of the original project.

---

**This code is a refactored version containing only the core functionality of the NMC system.**
