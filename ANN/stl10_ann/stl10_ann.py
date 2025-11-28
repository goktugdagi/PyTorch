"""
STL-10 Image Classification with a Fully-Connected Neural Network (ANN)
=======================================================================

This script trains a simple fully-connected (dense) neural network on the
STL-10 image classification dataset using PyTorch.

What it does:
-------------
1. Sets up device (CPU / GPU) and seeds for reproducibility
2. Downloads STL-10 dataset and applies:
   - Data augmentation for training
   - Normalization for validation/test
3. Splits train set into train/validation subsets (90% / 10%)
4. Builds a fully-connected ANN on flattened 96x96x3 RGB images
5. Trains with:
   - Mixed precision (if CUDA is available)
   - Learning-rate scheduler (StepLR)
   - Early stopping on validation accuracy
6. Saves:
   - Best model checkpoint (by validation accuracy)
   - Training history as CSV
   - Training curves (loss & accuracy) as PNGs
   - Confusion matrix for validation set as PNG
7. Prints:
   - Train/validation metrics during training
   - Final test results
   - Validation classification report (if scikit-learn is installed)

Run:
----
    python stl10_ann_fc.py

Make sure to install dependencies (see README / requirements.txt).
"""

import os
import csv
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, random_split


# -------------------------------------------------
# 0. Global Config
# -------------------------------------------------
# Output directory for model checkpoints, logs, and plots
OUTPUT_DIR = "outputs_stl10_ann"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training hyperparameters and paths
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 5
MIN_DELTA = 0.001
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
STEP_SIZE = 5
GAMMA = 0.5

CKPT_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
HISTORY_CSV = os.path.join(OUTPUT_DIR, "training_history.csv")
LOSS_PLOT = os.path.join(OUTPUT_DIR, "loss_curves.png")
ACC_PLOT = os.path.join(OUTPUT_DIR, "accuracy_curves.png")
CONFMAT_PNG = os.path.join(OUTPUT_DIR, "confmat_val.png")


# -------------------------------------------------
# 1. Device & Reproducibility
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# -------------------------------------------------
# 2. Dataset Statistics & Transforms
# -------------------------------------------------
# Pre-computed mean and std for STL-10 (RGB order)
STL_MEAN = [0.447412, 0.427196, 0.386978]
STL_STD = [0.260340, 0.256438, 0.271819]

# Data augmentation for training set
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
    ),
    transforms.ToTensor(),
    transforms.Normalize(STL_MEAN, STL_STD),
])

# Evaluation transform (no augmentation, only normalization)
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(STL_MEAN, STL_STD),
])


# -------------------------------------------------
# 3. Dataset Loading (STL-10)
# -------------------------------------------------
def load_stl10_datasets(data_root: str = "./data"):
    """
    Download and prepare STL-10 train/val/test datasets with
    augmentation for training and clean transforms for eval.
    """
    # Augmented version of the train split (for training)
    trainval_aug = datasets.STL10(
        root=data_root,
        split="train",
        download=True,
        transform=train_transform,
    )

    # Clean version of the same train split (for validation)
    full_val_clean = datasets.STL10(
        root=data_root,
        split="train",
        download=True,
        transform=eval_transform,
    )

    # Test split
    test_dataset = datasets.STL10(
        root=data_root,
        split="test",
        download=True,
        transform=eval_transform,
    )

    # Quick sanity check: one sample
    image, label = trainval_aug[1]

    class_names = trainval_aug.classes
    print(f"Class names: {class_names}")

    # More conventional mapping: class_name -> index
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    print(f"Class to index mapping: {class_to_idx}")

    print(
        f"trainval_aug length: {len(trainval_aug)}\n"
        f"full_val_clean length: {len(full_val_clean)}\n"
        f"test_dataset length: {len(test_dataset)}"
    )

    print(f"Image shape: {image.shape} -> [channels, height, width]")
    print(f"Image label: {class_names[label]}")

    # De-normalize and show one example image (for sanity check)
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(STL_MEAN, STL_STD)],
        std=[1 / s for s in STL_STD],
    )
    img_denorm = inv_normalize(image).clamp(0, 1)

    plt.imshow(img_denorm.permute(1, 2, 0))
    plt.title(class_names[label])
    plt.axis("off")
    plt.show()

    return trainval_aug, full_val_clean, test_dataset, class_names


# -------------------------------------------------
# 4. Train / Validation Split & DataLoaders
# -------------------------------------------------
def create_dataloaders(trainval_aug, full_val_clean, test_dataset):
    """
    Split train dataset into train/val and build DataLoaders.
    Validation uses the 'clean' version (no augmentation).
    """
    g = torch.Generator().manual_seed(SEED)
    n_total = len(trainval_aug)
    n_val = int(n_total * 0.10)   # 10% for validation
    n_train = n_total - n_val

    # Split augmented dataset into train & val indices
    train_subset, val_subset_augref = random_split(
        trainval_aug,
        [n_train, n_val],
        generator=g,
    )

    # Use the same indices on the "clean" version for validation
    val_subset = Subset(full_val_clean, val_subset_augref.indices)

    NUM_WORKERS = 0  # Set >0 if your environment supports multiprocessing well
    PIN_MEM = (device.type == "cuda")

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=False,
    )

    print(
        f"Split -> train: {len(train_subset)} | "
        f"val: {len(val_subset)} | "
        f"test: {len(test_dataset)}"
    )
    print(
        "Dataloaders ready:",
        f"\n  - batch_size={BATCH_SIZE}",
        f"\n  - num_workers={NUM_WORKERS}",
        f"\n  - pin_memory={PIN_MEM}",
    )

    return train_loader, val_loader, test_loader


# -------------------------------------------------
# 5. Helper Functions (accuracy, evaluation loops)
# -------------------------------------------------
def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute classification accuracy in percentage.
    """
    preds = y_pred.argmax(dim=1)
    return (preds == y_true).float().mean() * 100.0


# Mixed precision scaler (only active on CUDA)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn,
    optimizer,
    accuracy_fn,
    device=device,
    scaler=scaler,
):
    """
    One full training epoch over `data_loader` using mixed precision.
    Returns average loss and accuracy for the epoch.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward & loss
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        correct += (y_pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / len(data_loader)
    acc_pct = (correct / total) * 100.0

    return avg_loss, acc_pct


@torch.inference_mode()
def test_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn,
    accuracy_fn,
    device=device,
    use_amp: bool = True,
):
    """
    Evaluation loop (for validation / test).
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        running_loss += loss.item()
        correct += (y_pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / len(data_loader)
    acc_pct = (correct / total) * 100.0

    return avg_loss, acc_pct


def eval_model(
    model: nn.Module,
    data_loader_or_dataset,
    loss_fn,
    device=device,
):
    """
    Evaluate a model on a given DataLoader (or Dataset).
    Returns a small dict with model name, average loss and accuracy.
    """
    from torch.utils.data import Dataset  # local import to avoid circulars

    # Allow passing a raw Dataset instead of a DataLoader
    if isinstance(data_loader_or_dataset, Dataset):
        pin = (device.type == "cuda")
        data_loader = DataLoader(
            data_loader_or_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
            persistent_workers=False,
        )
    else:
        data_loader = data_loader_or_dataset

    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            running_loss += loss.item()
            correct += (y_pred.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / len(data_loader)
    acc_pct = (correct / total) * 100.0

    return {
        "model_name": model.__class__.__name__,
        "model_loss": avg_loss,
        "model_acc": acc_pct,
    }


# -------------------------------------------------
# 6. EarlyStopping Utility
# -------------------------------------------------
class EarlyStopping:
    """
    Early stopping on a single monitored metric (e.g. validation accuracy).

    Args:
        patience: number of epochs to wait without improvement
        min_delta: minimum change to qualify as an improvement
        mode: "max" (e.g. accuracy) or "min" (e.g. loss)
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.bad = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Take one step with the current monitored value.
        Returns True if there was an improvement.
        """
        if self.mode == "max":
            improved = value > (self.best + self.min_delta)
        else:
            improved = value < (self.best - self.min_delta)

        if improved:
            self.best = value
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.should_stop = True

        return improved


# -------------------------------------------------
# 7. ANN Model Definition
# -------------------------------------------------
class ANN(nn.Module):
    """
    Simple feed-forward fully-connected network for flattened 96x96x3 images.
    """

    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        dropout: float = 0.30,
    ):
        super(ANN, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_shape, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_units, hidden_units // 2)
        self.bn2 = nn.BatchNorm1d(hidden_units // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_units // 2, hidden_units // 4)
        self.bn3 = nn.BatchNorm1d(hidden_units // 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(hidden_units // 4, output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x


# -------------------------------------------------
# 8. Confusion Matrix utilities
# -------------------------------------------------
def collect_preds(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Collect all predictions and labels from a given DataLoader.
    Returns: (y_true, y_pred) as numpy arrays.
    """
    model.eval()
    all_p, all_y = [], []

    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            all_p.append(logits.argmax(dim=1).cpu().numpy())
            all_y.append(y.numpy())

    return np.concatenate(all_y), np.concatenate(all_p)


def plot_confmat(
    cm: np.ndarray,
    labels,
    title: str,
    outfile: str,
    normalize: bool = True,
):
    """
    Plot and save a confusion matrix as a PNG file.
    """
    cm2 = cm.astype(np.float32)
    if normalize:
        # Row-normalize to percentages
        cm2 = cm2 / cm2.sum(axis=1, keepdims=True).clip(min=1e-9)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm2, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(outfile, dpi=150)
    plt.close()
    print("Saved:", outfile)


# -------------------------------------------------
# 9. Main training pipeline
# -------------------------------------------------
def main():
    # 1) Load STL-10 datasets
    trainval_aug, full_val_clean, test_dataset, class_names = load_stl10_datasets()
    num_classes = len(class_names)

    # 2) Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        trainval_aug,
        full_val_clean,
        test_dataset,
    )

    # 3) Define model, loss, optimizer, scheduler
    torch.manual_seed(SEED)

    INPUT_SHAPE = 3 * 96 * 96
    HIDDEN_UNITS = 1024

    model = ANN(
        input_shape=INPUT_SHAPE,
        hidden_units=HIDDEN_UNITS,
        output_shape=num_classes,
    ).to(device)

    print("Model parameters:")
    print(model)
    print("State dict keys:", list(model.state_dict().keys()))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA,
    )

    # Re-create scaler explicitly (clear intent for training section)
    global scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4) History dict to log metrics per epoch
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # Early stopper on validation accuracy
    early = EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        mode="max",
    )

    best_val = -1.0        # Track best validation accuracy
    best_state = None      # To hold best model weights on CPU

    # 5) Training loop
    for epoch in range(EPOCHS):

        tr_loss, tr_acc = train_step(
            model,
            train_loader,
            loss_fn,
            optimizer,
            accuracy_fn,
            device,
            scaler,
        )
        val_loss, val_acc = test_step(
            model,
            val_loader,
            loss_fn,
            accuracy_fn,
            device,
            use_amp=True,
        )

        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)

        print(
            f"Epoch {epoch + 1:02d} | LR: {lr_now:.5g} | "
            f"Train Loss: {tr_loss:.5f} | Train Acc: {tr_acc:.2f}% | "
            f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.2f}%"
        )

        # Save best checkpoint based on validation accuracy
        if val_acc > best_val:
            best_val = val_acc
            # Clone state dict to CPU tensors to avoid reference issues
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            torch.save(
                {
                    "state_dict": best_state,
                    "val_acc": best_val,
                    "epoch": epoch,
                },
                CKPT_PATH,
            )

        # Step the LR scheduler
        scheduler.step()

        # Early stopping step on validation accuracy
        if early.step(val_acc):
            # Improved this epoch, continue training
            pass
        elif early.should_stop:
            print(
                f"[EarlyStopping] No improvement for {PATIENCE} epochs. "
                f"Stopping at epoch {epoch + 1}."
            )
            break

    # 6) Load best weights before final evaluation & plots
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    # 7) Evaluate on test set with best weights
    model_results = eval_model(
        model=model,
        data_loader_or_dataset=test_loader,
        loss_fn=loss_fn,
        device=device,
    )
    print("Test set evaluation (best weights):", model_results)

    ts_loss, ts_acc = test_step(
        model,
        test_loader,
        loss_fn,
        accuracy_fn,
        device,
        use_amp=True,
    )
    print(
        f"\nFinal Test -> Loss: {ts_loss:.5f} | Acc: {ts_acc:.2f}%  "
        f"(best Val Acc: {best_val:.2f}%)"
    )
    print(f"Best checkpoint: {os.path.abspath(CKPT_PATH)}")

    # 8) Plot training curves (Loss & Accuracy)
    epochs_ran = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs_ran, history["train_loss"], label="Train Loss")
    plt.plot(epochs_ran, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_PLOT, dpi=150)
    plt.show()
    print("Saved:", LOSS_PLOT)

    plt.figure()
    plt.plot(epochs_ran, history["train_acc"], label="Train Acc (%)")
    plt.plot(epochs_ran, history["val_acc"], label="Val Acc (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(ACC_PLOT, dpi=150)
    plt.show()
    print("Saved:", ACC_PLOT)

    # 9) Save training history as CSV
    try:
        with open(HISTORY_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
            )
            for i in range(len(history["train_loss"])):

                writer.writerow(
                    [
                        i + 1,
                        history["train_loss"][i],
                        history["train_acc"][i],
                        history["val_loss"][i],
                        history["val_acc"][i],
                        history["lr"][i],
                    ]
                )
        print(f"History saved to: {os.path.abspath(HISTORY_CSV)}")
    except Exception as e:
        print("CSV save skipped:", e)

    # 10) Confusion Matrix & Classification Report (Validation)
    CLASS_NAMES = class_names  # alias

    y_true_val, y_pred_val = collect_preds(model, val_loader, device)

    # Try to import sklearn metrics (optional)
    try:
        from sklearn.metrics import confusion_matrix, classification_report

        SK_OK = True
    except Exception:
        SK_OK = False

    if SK_OK:
        cm_val = confusion_matrix(
            y_true_val,
            y_pred_val,
            labels=list(range(len(CLASS_NAMES))),
        )
    else:
        # Manual fallback if sklearn is not installed
        num = len(CLASS_NAMES)
        cm_val = np.zeros((num, num), dtype=np.int64)
        for t, p in zip(y_true_val, y_pred_val):
            cm_val[t, p] += 1

    plot_confmat(
        cm_val,
        CLASS_NAMES,
        "Confusion Matrix (Val)",
        CONFMAT_PNG,
        normalize=True,
    )

    if SK_OK:
        print(
            "\nClassification report (Val):\n",
            classification_report(
                y_true_val,
                y_pred_val,
                target_names=CLASS_NAMES,
                digits=3,
            ),
        )


if __name__ == "__main__":
    main()
