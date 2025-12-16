# Flowers102 Classification with EfficientNetB3 (PyTorch)

This project implements a **two-stage transfer learning pipeline** for multi-class image classification on the **Oxford Flowers102** dataset using **EfficientNetB3** and **PyTorch**.

The focus of this work is not only model accuracy, but also **training engineering best practices**, including freezing vs fine-tuning, mixed precision training, checkpointing, and proper evaluation.

---

## 🔍 Project Overview

- **Task:** Multi-class image classification (102 flower classes)
- **Dataset:** Oxford Flowers102
- **Model:** EfficientNetB3 (pretrained on ImageNet)
- **Framework:** PyTorch
- **Training Strategy:** Frozen backbone → Fine-tuning
- **Hardware Support:** CPU / GPU (CUDA with AMP)

---

## 🧠 Model & Training Strategy

### EfficientNetB3
EfficientNetB3 is a convolutional neural network that balances **depth, width, and input resolution** using compound scaling.  
It provides strong performance with relatively fewer parameters compared to traditional CNNs.

---

### Two-Stage Training

#### Stage 1 – Frozen Backbone
- The feature extractor (`model.features`) is frozen
- Only the classifier head is trained
- Purpose: allow the new classifier to adapt to the Flowers102 label space
- Learning Rate: `1e-3`

#### Stage 2 – Fine-Tuning
- The last **N EfficientNet blocks** are unfrozen
- Classifier remains trainable
- Purpose: adapt high-level visual features to flower-specific patterns
- Reduced Learning Rate: `1e-4`

```python
UNFREEZE_LAST_BLOCKS = 2
```

---

## ⚙️ Configuration

Key hyperparameters used in this project:

```python
IMG_SIZE = (300, 300)
BATCH_SIZE = 32

EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 5

LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4
```

---

## 🧪 Data Preprocessing & Augmentation

### Training Transforms
- Resize to `300×300`
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization with ImageNet mean & std

### Validation Transforms
- Resize
- Normalization only (no augmentation)

---

## ⚡ Mixed Precision Training (AMP)

- Automatic Mixed Precision is enabled when CUDA is available
- Implemented using:
  - `torch.amp.autocast`
  - `torch.amp.GradScaler`
- Provides faster training and reduced GPU memory usage

---

## 📈 Evaluation

The model is evaluated on the official **validation split** of Flowers102.

Evaluation includes:
- Validation accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix visualization

> Note: Each class contains only **10 validation images**, which causes high variance in per-class metrics.

### Typical Results
- Validation Accuracy: ~75%
- Macro F1-score: ~0.74

---

## 💾 Checkpointing & Outputs

During training, the following artifacts are saved automatically:

```
runs/
 ├── effb3_flowers102_finetune_best.pth   # Best model (highest val accuracy)
 ├── effb3_flowers102_finetune_last.pth   # Last epoch checkpoint
 ├── effb3_flowers102_finetune_loss.png   # Train / Val loss curves
 └── effb3_flowers102_finetune_acc.png    # Train / Val accuracy curves
```

The **best checkpoint** is selected based on validation accuracy.

---

## ▶️ How to Run

1. Install dependencies:
```bash
pip install torch torchvision matplotlib seaborn scikit-learn tqdm
```

2. Run training:
```bash
python efficientnet_b3.py
```

> The Flowers102 dataset will be downloaded automatically on first run.

---

## 📌 Notes

- The code is written to be **reproducible and readable**
- Designed as a **portfolio-quality transfer learning example**
- Suitable as a reference for:
  - Frozen vs fine-tuning strategies
  - EfficientNet-based classification
  - Mixed precision training in PyTorch

---

## 👤 Author

**Göktuğ**  
Machine Learning / Computer Vision
