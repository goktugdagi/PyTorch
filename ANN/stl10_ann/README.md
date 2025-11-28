# STL-10 Image Classification with Fully-Connected Neural Network (ANN)

This project trains a simple fully-connected (dense) neural network on the
[STL-10 dataset](https://cs.stanford.edu/~acoates/stl10/) using PyTorch.

The goal is **not** to beat SOTA, but to:

- Practice end-to-end ANN training on a real image dataset
- Use **data augmentation**, **mixed precision**, **LR scheduling** and **early stopping**
- Log metrics, save best checkpoints, and visualize results

> **File:** `stl10_ann.py`  
> **Folder:** `PyTorch/ANN/stl10_ann/`

---

## Project structure

```text
stl10_ann/
├── stl10_ann.py      # Main training script
├── README.md            # This file
└── requirements.txt     # Python dependencies (optional but recommended)
