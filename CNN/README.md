# FashionMNIST CNN Classifier 🧥👟👗

This repository contains a **Convolutional Neural Network (CNN)** implemented in **PyTorch** to classify the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  
The script is fully commented in English and includes training, validation, and testing workflows with additional features.

---

## 🚀 Features
- **Data Augmentation**: Random crop, rotation, and horizontal flip for better generalization.  
- **CNN Architecture**: 3 convolutional blocks + batch normalization + dropout regularization.  
- **Training Utilities**:
  - Automatic Mixed Precision (AMP) for faster training on GPU
  - EarlyStopping (patience-based)
  - Learning rate scheduler
  - Best model checkpoint saving
- **Evaluation**:
  - Accuracy & loss tracking
  - Confusion matrix plotting
  - Classification report (requires scikit-learn)

---

## 📦 Requirements
Make sure you have Python 3.8+ installed.  
Install the dependencies with:

```bash
pip install torch torchvision matplotlib scikit-learn tqdm
```

---

## ▶️ Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/goktugdagi/PyTorch.git
   cd PyTorch/CNN
   ```

2. Run the training script:
   ```bash
   python fashionmnist_commented.py
   ```

3. Training outputs:
   - `best_model.pt` → saved best checkpoint  
   - `training_history.csv` → CSV log of training/validation metrics  
   - `confmat_val.png` → Confusion matrix plot  

---

## 📊 Results
- Training and validation curves are displayed at the end of training.  
- Confusion matrix and classification report show per-class performance.  

---

## 📂 Repository Structure
```
fashionmnist-cnn/
│
├── fashionmnist_commented.py   # Main training & evaluation script
├── training_history.csv        # Auto-generated after training
├── best_model.pt               # Saved best model checkpoint
├── confmat_val.png             # Confusion matrix (validation set)
└── README.md                   # Project description & usage guide
```

---

## 📝 Notes
- The script automatically detects and uses **GPU (CUDA)** if available.
- Training may take ~5–10 minutes on GPU depending on hardware.  
- Early stopping prevents overfitting and saves computation time.  

---

## 📜 License
This project is released under the MIT License.
