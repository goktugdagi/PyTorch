import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Config
# =========================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (device.type == "cuda")

IMG_SIZE = (300, 300)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
NUM_CLASSES = 102

EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 5
UNFREEZE_LAST_BLOCKS = 2

LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4

RUN_NAME = "effb3_flowers102_finetune"
OUT_DIR = "runs"
os.makedirs(OUT_DIR, exist_ok=True)

BEST_CKPT_PATH = os.path.join(OUT_DIR, f"{RUN_NAME}_best.pth")
LAST_CKPT_PATH = os.path.join(OUT_DIR, f"{RUN_NAME}_last.pth")
PLOT_LOSS_PATH = os.path.join(OUT_DIR, f"{RUN_NAME}_loss.png")
PLOT_ACC_PATH  = os.path.join(OUT_DIR, f"{RUN_NAME}_acc.png")

# =========================
# Transforms
# =========================
transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

transform_val = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# =========================
# Dataset / Loader
# =========================
train_dataset = datasets.Flowers102("./data", split="train", download=True, transform=transform_train)
val_dataset   = datasets.Flowers102("./data", split="val", download=True, transform=transform_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# =========================
# Model
# =========================
weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
model = models.efficientnet_b3(weights=weights)

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler(enabled=USE_AMP)

# =========================
# Helpers
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss_sum += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total

def save_ckpt(path, epoch, model, optimizer, scheduler, best_acc, history):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc,
        "history": history
    }, path)

def unfreeze_last_blocks(model, n):
    for p in model.features.parameters():
        p.requires_grad = False
    if n > 0:
        for block in list(model.features.children())[-n:]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

# =========================
# Training
# =========================
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = -1.0
global_epoch = 0

# Stage 1 – Frozen
for p in model.features.parameters():
    p.requires_grad = False

optimizer = optim.Adam(model.classifier.parameters(), lr=LR_STAGE1)
scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

for _ in range(EPOCHS_STAGE1):
    global_epoch += 1
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"[S1] Epoch {global_epoch}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_loss, train_acc = total_loss / total, correct / total
    val_loss, val_acc = evaluate(model, val_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    save_ckpt(LAST_CKPT_PATH, global_epoch, model, optimizer, scheduler, best_val_acc, history)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_ckpt(BEST_CKPT_PATH, global_epoch, model, optimizer, scheduler, best_val_acc, history)

# Stage 2 – Fine-tuning
unfreeze_last_blocks(model, UNFREEZE_LAST_BLOCKS)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE2)
scheduler = optim.lr_scheduler.StepLR(optimizer, 3, 0.1)

for _ in range(EPOCHS_STAGE2):
    global_epoch += 1
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"[S2] Epoch {global_epoch}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_loss, train_acc = total_loss / total, correct / total
    val_loss, val_acc = evaluate(model, val_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    save_ckpt(LAST_CKPT_PATH, global_epoch, model, optimizer, scheduler, best_val_acc, history)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_ckpt(BEST_CKPT_PATH, global_epoch, model, optimizer, scheduler, best_val_acc, history)

# =========================
# Final Evaluation
# =========================
ckpt = torch.load(BEST_CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

all_preds, all_labels = [], []
with torch.inference_mode():
    for images, labels in val_loader:
        images = images.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            outputs = model(images)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, digits=4))
