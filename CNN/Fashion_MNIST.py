# --------------------
# Imports & Utilities
# --------------------
import torch                              # Core PyTorch library for tensors and deep learning
import torchvision                         # Vision utilities (datasets, models, transforms)
import random, os, csv                     # random for reproducibility; os/csv for paths & logging
import numpy as np                         # Numerical computing utilities
from torch import nn, optim                # Neural network modules and optimizers
from torch.utils.data import DataLoader, random_split, Subset  # Data loading & splitting helpers
from torchvision import datasets, transforms                    # Datasets and image transforms
from tqdm.auto import tqdm                 # (Optional) progress bars; not essential here
from torchvision.transforms import ToTensor # Utility to convert PIL images to tensors
import matplotlib.pyplot as plt            # Plotting for sample visualization and curves
import math                                # Math helpers (used in EarlyStopping)

# ------------------
# Version & Device
# ------------------
print(torch.__version__)                   # Print PyTorch version for debugging/reproducibility
print(torchvision.__version__)             # Print torchvision version for debugging

try:
    device                                 # Try to use an existing 'device' if defined upstream
except NameError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto-select device
print("Device:", device)                  # Display selected device (cpu/cuda)

# =========================
# 0) Import + Device + Seed
# =========================
SEED = 42                                  # Reproducibility seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)  # Seed Python, NumPy, and PyTorch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # (kept from original but unused)
if device.type == "cuda":                # If CUDA is available
    torch.cuda.manual_seed_all(SEED)       # Seed all CUDA devices
print("Device:", device)                  # Re-confirm device (mirrors original)

# -----------------------------------------------------------
# 1) Transforms & Datasets (train-only augment) + train/val split
#    (val/test use clean transforms)
# -----------------------------------------------------------
FMNIST_MEAN, FMNIST_STD = (0.2860,), (0.3530,)   # Precomputed channel-wise mean/std for FashionMNIST

# Augmentations applied ONLY to training data to improve generalization
train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=2),        # Slight random cropping to add translation invariance
    transforms.RandomRotation(10),               # Random small rotations
    transforms.RandomHorizontalFlip(p=0.5),      # Flip left-right with probability 0.5
    transforms.ToTensor(),                       # Convert PIL image to torch.Tensor in [0,1]
    transforms.Normalize(FMNIST_MEAN, FMNIST_STD), # Normalize with dataset stats
])

# Clean (deterministic) transform for validation and test
eval_transform = transforms.Compose([
    transforms.ToTensor(),                       # Convert to tensor
    transforms.Normalize(FMNIST_MEAN, FMNIST_STD), # Normalize (no augmentation)
])

# Train+val source dataset with augmentations (only used to draw samples for both splits)
trainval_aug = datasets.FashionMNIST(
    root="./data",                             # Local data folder
    train=True,                                  # Use training split of the dataset
    download=True,                               # Download if not present
    transform=train_transform                    # Apply augmentation pipeline
)

# A separate copy pointing to the same underlying data but with CLEAN transforms
full_val_clean = datasets.FashionMNIST(
    root="./data",                             # Same data root
    train=True,                                  # Same training split for indexing consistency
    download=True,                               # Ensure downloaded
    transform=eval_transform                     # Clean transform for validation subset
)

# Test dataset with CLEAN transforms
test_dataset  = datasets.FashionMNIST(
    root="./data",                             # Data root
    train=False,                                 # Use test split
    download=True,                               # Download if needed
    transform=eval_transform                     # Clean transform
)


image, label = trainval_aug[0]                   # Fetch a single augmented sample/label
print(label)                                           # (Notebook artifact) evaluates to label in interactive cells
print(image)                                           # (Notebook artifact) evaluates to image tensor in interactive cells

class_names = trainval_aug.classes               # Human-readable class names (list of 10 categories)
print(class_names)                                      # (Notebook artifact) display class names

class_to_idx = trainval_aug.class_to_idx         # Mapping from class name to integer index
print(class_to_idx)                                    # (Notebook artifact) display mapping

trainval_aug.targets                             # Tensor of target labels for the train split
print(len(trainval_aug), len(full_val_clean), len(test_dataset))  # (Notebook artifact) check dataset sizes

# Print shapes/labels and visualize one example
print(f"Image shape: {image.shape} -> [color_channels, height, width]")   # Confirm tensor shape
print(f"Image label: {class_names[label]}")                               # Show label name
plt.imshow(image.squeeze(), cmap="gray")                                  # Plot the image in grayscale
plt.title(class_names[label])                                               # Title with class name
plt.axis("off");                                                          # Turn off axes

# Plot a 4x4 grid of random training images
torch.manual_seed(42)                                                       # Seed for reproducible selection
fig = plt.figure(figsize=(9, 9))                                            # Prepare figure
rows, cols = 4, 4                                                           # Grid size
for i in range(1, rows * cols + 1):                                         # Iterate over 16 subplots
  random_idx = torch.randint(0, len(trainval_aug), size=[1]).item()         # Random index
  img, label = trainval_aug[random_idx]                                     # Sample (with augmentations)
  fig.add_subplot(rows, cols, i)                                            # Add subplot
  plt.imshow(img.squeeze(), cmap="gray")                                  # Show image
  plt.title(class_names[label])                                             # Subplot title
  plt.axis("off");                                                        # Hide axes

# ------------------
# Train/Val Splits
# ------------------
g = torch.Generator().manual_seed(SEED)                                     # Generator for deterministic split
n_total = len(trainval_aug)                                                 # Total number of train+val samples
n_val = int(n_total * 0.10)                                                 # 10% for validation
n_train = n_total - n_val                                                   # Remaining for training

train_subset, val_subset_augref = random_split(                             # Random split using augment dataset
    trainval_aug, [n_train, n_val], generator=g
)

val_subset = Subset(full_val_clean, val_subset_augref.indices)              # Validation subset with CLEAN transform

# -----------------
# DataLoaders
# -----------------
BATCH_SIZE = 128                                                            # Mini-batch size
num_workers = 2                                                             # Background workers for data loading
pin_mem = (device.type == "cuda")                                         # Faster host→GPU transfers if CUDA
# CUDA note: pin_memory=True helps for GPU; persistent_workers=True valid when num_workers>0

train_loader = DataLoader(                                                  # Training loader
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,                                                           # Shuffle for SGD
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True
)

val_loader   = DataLoader(                                                  # Validation loader (no shuffle)
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True
)

test_loader  = DataLoader(                                                  # Test loader (no shuffle)
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True
)

print(f"Split -> train:{len(train_subset)} | val:{len(val_subset)} | test:{len(test_dataset)}")  # Summary sizes

print("Dataloaders ready:",                                                # Display dataloader configuration
      f"\n  - batch_size={BATCH_SIZE}",
      f"\n  - num_workers={num_workers}",
      f"\n  - pin_memory={pin_mem}",
      )

# ---------------------
# 2) Helper Functions
# ---------------------

def accuracy_fn(y_true, y_pred):                                            # Accuracy metric (percentage)
    # If you want ratio instead of percentage, divide by 100 outside
    preds = y_pred.argmax(dim=1)                                            # Predicted class indices
    return (preds == y_true).float().mean() * 100.0                         # Mean correctness × 100

# AMP scaler: enabled only on CUDA for mixed-precision speedups
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))       # Handles loss scaling under AMP

# --------
# Train
# --------
def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device=device, scaler=scaler):
    model.train()                                                           # Put model in training mode
    running_loss, correct, total = 0.0, 0, 0                                # Stats accumulators

    for X, y in data_loader:                                                # Loop over mini-batches
        X, y = X.to(device), y.to(device)                                   # Move data to device

        optimizer.zero_grad(set_to_none=True)                               # Reset gradients efficiently

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):          # Enable mixed precision on CUDA
            y_pred = model(X)                                               # Forward pass
            loss = loss_fn(y_pred, y)                                       # Compute loss

        scaler.scale(loss).backward()                                       # Backprop with scaled loss
        scaler.step(optimizer)                                              # Optimizer step (unscales internally)
        scaler.update()                                                     # Update scaler for next iter

        running_loss += loss.item()                                         # Accumulate batch loss
        correct += (y_pred.argmax(dim=1) == y).sum().item()                 # Accumulate #correct
        total   += y.size(0)                                                # Accumulate #samples

    avg_loss = running_loss / len(data_loader)                              # Average loss over batches
    acc_pct  = (correct / total) * 100.0                                    # Accuracy in percent

    return avg_loss, acc_pct                                                # Return epoch stats

# --------
# Test/Val
# --------

@torch.inference_mode()                                                     # Disable grad for efficiency
def test_step(model, data_loader, loss_fn, accuracy_fn, device=device, use_amp=True):
    model.eval()                                                            # Eval mode (BN/Dropout frozen)
    running_loss, correct, total = 0.0, 0, 0                                # Reset stats

    for X, y in data_loader:                                                # Iterate over batches
        X, y = X.to(device), y.to(device)                                   # Move to device

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type=="cuda")):  # AMP if CUDA
            y_pred = model(X)                                               # Forward
            loss   = loss_fn(y_pred, y)                                     # Loss

        running_loss += loss.item()                                         # Accumulate loss
        correct += (y_pred.argmax(dim=1) == y).sum().item()                 # Accumulate correct
        total   += y.size(0)                                                # Accumulate total

    avg_loss = running_loss / len(data_loader)                              # Avg loss
    acc_pct  = (correct / total) * 100.0                                    # Accuracy %

    return avg_loss, acc_pct                                                # Return stats

# -------------------------
# Val helper (safe input)
# -------------------------

def eval_model(model, data_loader, loss_fn, accuracy_fn, device=device):
    from torch.utils.data import Dataset                                     # Local import (optional)
    if isinstance(data_loader, Dataset):                                     # Accept a Dataset by mistake
        # If a raw Dataset is provided, wrap it in a DataLoader automatically
        pin = (device.type=="cuda")
        data_loader = DataLoader(data_loader, batch_size=128, shuffle=False,
                                 num_workers=2, pin_memory=pin, persistent_workers=True)
    model.eval()                                                             # Eval mode
    running_loss, correct, total = 0.0, 0, 0                                 # Stat accumulators
    with torch.inference_mode():                                             # No grad
        for X, y in data_loader:                                             # Iterate
            X, y = X.to(device), y.to(device)                                # To device
            y_pred = model(X)                                                # Forward
            loss   = loss_fn(y_pred, y)                                      # Loss
            running_loss += loss.item()                                      # Accumulate
            correct += (y_pred.argmax(dim=1) == y).sum().item()              # Correct
            total   += y.size(0)                                             # Total
    avg_loss = running_loss / len(data_loader)                               # Average loss
    acc_pct  = (correct / total) * 100.0                                     # Accuracy %
    return {"model_name": model.__class__.__name__,                        # Return dict summary
            "model_loss": avg_loss, "model_acc": acc_pct}

# ---------------------------------------
# EarlyStopping utility (patience-based)
# ---------------------------------------

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="max"):
        self.patience = patience; self.min_delta = min_delta; self.mode = mode  # Config
        self.best = -math.inf if mode=="max" else math.inf                    # Init best based on mode
        self.bad  = 0; self.should_stop = False                                 # Counters/flag

    def step(self, value):
        # Determine if 'value' improved enough relative to 'best'
        improved = (value > self.best + self.min_delta) if self.mode=="max" else (value < self.best - self.min_delta)
        if improved:                                                            # If improved
            self.best = value; self.bad = 0                                     # Update best & reset patience counter
        else:                                                                   # Otherwise increment patience
            self.bad += 1
            if self.bad >= self.patience:                                       # If patience exceeded
                self.should_stop = True                                         # Signal to stop
        return improved                                                         # Return whether improved


# ----------------
# 3) CNN Model
# ----------------

class CNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, dropout: float=0.30):
        super().__init__()                                                      # Initialize nn.Module
        # Block 1: two convs + BN + ReLU → MaxPool (28→14)
        self.block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1, bias=False),    # 3×3 conv
            nn.BatchNorm2d(hidden_units),                                       # BN for stable training
            nn.ReLU(inplace=True),                                              # Nonlinearity
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1, bias=False),    # 3×3 conv
            nn.BatchNorm2d(hidden_units),                                       # BN
            nn.ReLU(inplace=True),                                              # ReLU
            nn.MaxPool2d(2)                                                     # Downsample H,W by 2
        )
        # Block 2: channels×2 then pool (14→7)
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1, bias=False),  # Expand channels
            nn.BatchNorm2d(hidden_units*2),                                     # BN
            nn.ReLU(inplace=True),                                              # ReLU
            nn.Conv2d(hidden_units*2, hidden_units*2, 3, padding=1, bias=False),# Keep channels
            nn.BatchNorm2d(hidden_units*2),                                     # BN
            nn.ReLU(inplace=True),                                              # ReLU
            nn.MaxPool2d(2)                                                     # Downsample
        )
        # Block 3: final conv to deepen features (7×7 stays 7×7)
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_units*2, hidden_units*4, 3, padding=1, bias=False),# More channels
            nn.BatchNorm2d(hidden_units*4),                                     # BN (optional but helpful)
            nn.ReLU(inplace=True)                                               # ReLU
        )
        self.pool = nn.AdaptiveAvgPool2d(1)                                     # Global average pool → (C,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                                       # Flatten to (N,C)
            nn.Dropout(p=dropout),                                              # Regularization
            nn.Linear(hidden_units*4, output_shape)                             # Final logits (10 classes)
        )
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)              # Stacked conv blocks
        x = self.pool(x);   x = self.classifier(x)                              # GAP → classifier
        return x                                                                # Return logits


# Instantiate the CNN model and move to device
torch.manual_seed(42)                                                           # Seed for weight init reproducibility
model = CNN(input_shape=1, hidden_units=32, output_shape=10).to(device)        # 1-channel grayscale input

print(model.state_dict())                                                               # (Notebook artifact) view parameters

# -----------------------------
# Loss / Optimizer / Scheduler
# -----------------------------

cnn_loss_fn   = nn.CrossEntropyLoss()                                           # Multiclass classification loss
cnn_optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)      # Adam + L2 weight decay
cnn_scheduler = optim.lr_scheduler.StepLR(cnn_optimizer, step_size=5, gamma=0.5)# Step LR scheduler
scaler    = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))          # AMP scaler (again for clarity)

# ---------------
# Training Setup
# ---------------

EPOCHS = 30                                                                     # Max epochs
PATIENCE = 5                                                                    # Early stopping patience
MIN_DELTA = 0.001                                                               # Minimum improvement to count
CKPT_PATH = "best_model.pt"                                                    # Where to save best checkpoint
HISTORY_CSV = "training_history.csv"                                          # Where to store training curves

history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[], "lr":[]}  # Keep metrics
early = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, mode="max")     # Early stopper on val acc
best_val = -1.0                                                                 # Track best val accuracy
best_state = None                                                               # Snapshot of the best model state

for epoch in range(1, EPOCHS+1):                                                # Epoch loop
    tr_loss, tr_acc = train_step(model, train_loader, cnn_loss_fn, cnn_optimizer, accuracy_fn, device, scaler)  # Train
    vl_loss, vl_acc = test_step(model,  val_loader,   cnn_loss_fn, accuracy_fn, device, use_amp=True)           # Validate

    lr_now = cnn_optimizer.param_groups[0]["lr"]                               # Read current LR
    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)  # Log train stats
    history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)    # Log val stats
    history["lr"].append(lr_now)                                                 # Log LR

    print(f"Epoch {epoch:02d} | LR:{lr_now:.5g} | "                           # Pretty epoch log
          f"Train Loss:{tr_loss:.5f} | Train Acc:{tr_acc:.2f}% | "
          f"Val Loss:{vl_loss:.5f} | Val Acc:{vl_acc:.2f}%")

    # ---- BEST CHECKPOINT (fixed line) ----
    if vl_acc > best_val:                                                       # If validation accuracy improved
        best_val = vl_acc                                                       # Update best
        # Clone state dict to CPU tensors to avoid reference issues
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"state_dict": best_state, "val_acc": best_val, "epoch": epoch}, CKPT_PATH)  # Save checkpoint

    cnn_scheduler.step()                                                        # Step the LR scheduler

    # Early stopping step on validation accuracy
    if early.step(vl_acc):                                                      # If improved, do nothing else
        pass
    elif early.should_stop:                                                     # If patience exhausted
        print(f"[EarlyStopping] No improvement for {PATIENCE} epochs. Stopping at epoch {epoch}.")
        break

# ----------
# Evaluate
# ----------

model_results = eval_model(                                                     # Evaluate on test loader
    model=model,
    data_loader=test_loader,
    loss_fn=cnn_loss_fn,
    accuracy_fn=accuracy_fn,
    device=device
)

print(model_results)                                                                    # (Notebook artifact) show dict

# Load the best model state (if any) before final plots/reports
if best_state is not None:
    model.load_state_dict(best_state)                                           # Restore best weights
model.to(device)                                                                 # Ensure on correct device

# -------------------------------
# Plot training curves (loss/acc)
# -------------------------------

epochs_ran = range(1, len(history["train_loss"]) + 1)                          # x-axis epochs actually run

plt.figure()                                                                     # New figure for Loss
plt.plot(epochs_ran, history["train_loss"], label="Train Loss")              # Plot train loss
plt.plot(epochs_ran, history["val_loss"],   label="Val Loss")                # Plot val loss
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch"); plt.legend(); plt.grid(True)  # Styling
plt.show()                                                                       # Display

plt.figure()                                                                     # New figure for Accuracy
plt.plot(epochs_ran, history["train_acc"], label="Train Acc (%)")            # Plot train accuracy
plt.plot(epochs_ran, history["val_acc"],   label="Val Acc (%)")              # Plot val accuracy
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy vs Epoch"); plt.legend(); plt.grid(True)
plt.show()                                                                       # Display

ts_loss, ts_acc = test_step(model, test_loader, cnn_loss_fn, accuracy_fn, device, use_amp=True)  # Final test
print(f"\nLoss:{ts_loss:.5f} | Acc:{ts_acc:.2f}%  (best Val Acc:{best_val:.2f}%)")             # Summary line
print(f"Best checkpoint: {os.path.abspath(CKPT_PATH)}")                                         # Absolute path

# ------------------------------
# Save history CSV to disk
# ------------------------------

try:
    with open(HISTORY_CSV, "w", newline="") as f:                             # Open CSV file
        w = csv.writer(f)                                                        # Create writer
        w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])  # Header
        for i in range(len(history["train_loss"])):                             # Iterate rows
            w.writerow([i+1, history["train_loss"][i], history["train_acc"][i],
                        history["val_loss"][i], history["val_acc"][i], history["lr"][i]])
    print(f"History saved to: {os.path.abspath(HISTORY_CSV)}")                  # Confirmation
except Exception as e:
    print("CSV save skipped:", e)                                              # Graceful fallback

# -------------------------------------------------------------
# 4) Confusion Matrix + Classification Report (Val/Test)
# -------------------------------------------------------------

CLASS_NAMES = trainval_aug.classes                                              # Class names again for reports

def collect_preds(model, loader, device):
    model.eval(); all_p, all_y = [], []                                         # Storage for preds/labels
    with torch.inference_mode():                                                # No grads
        for X, y in loader:                                                     # Iterate over loader
            X = X.to(device)                                                    # To device (y kept on CPU for concat)
            logits = model(X)                                                   # Forward pass
            all_p.append(logits.argmax(dim=1).cpu().numpy())                    # Predicted class indices (CPU numpy)
            all_y.append(y.numpy())                                             # True labels as numpy
    return np.concatenate(all_y), np.concatenate(all_p)                          # Return stacked arrays

# Try to import sklearn metrics (optional)

try:
    from sklearn.metrics import confusion_matrix, classification_report         # Confusion matrix & report
    SK_OK = True
except Exception:
    SK_OK = False

def plot_confmat(cm, labels, title, outfile, normalize=True):
    import matplotlib.pyplot as plt                                             # Local import is fine
    cm2 = cm.astype(np.float32)                                                # Work on float copy
    if normalize:
        cm2 = cm2 / cm2.sum(axis=1, keepdims=True).clip(min=1e-9)              # Row-normalize to percentages
    plt.figure(figsize=(7,6))                                                   # Figure size
    plt.imshow(cm2, interpolation="nearest")                                  # Heatmap
    plt.title(title)                                                            # Title
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")           # X tick labels
    plt.yticks(range(len(labels)), labels)                                      # Y tick labels
    plt.colorbar(fraction=0.046, pad=0.04)                                      # Colorbar
    plt.tight_layout(); plt.xlabel("Predicted"); plt.ylabel("True")          # Layout & axis labels
    plt.savefig(outfile, dpi=150); plt.close()                                  # Save to file and close figure
    print("Saved:", outfile)                                                   # Log saved path

# Compute VAL confusion matrix/report
y_true_val, y_pred_val = collect_preds(model, val_loader, device)               # Collect val predictions
if SK_OK:
    cm_val = confusion_matrix(y_true_val, y_pred_val, labels=list(range(len(CLASS_NAMES))))  # Confusion matrix
else:
    num = len(CLASS_NAMES); cm_val = np.zeros((num, num), dtype=np.int64)       # Manual fallback if sklearn missing
    for t,p in zip(y_true_val, y_pred_val): cm_val[t,p] += 1                    # Increment cells
plot_confmat(cm_val, CLASS_NAMES, "Confusion Matrix (Val)", "confmat_val.png", normalize=True)  # Save plot
if SK_OK:
    print("\nClassification report:\n",                                     # Print sklearn report if available
          classification_report(y_true_val, y_pred_val, target_names=CLASS_NAMES, digits=3))
