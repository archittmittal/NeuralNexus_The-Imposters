import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from collections import Counter
from PIL import Image

# ============================================================================== #
# Configuration & Setup
# ============================================================================== #
# Automatically adjust path if running in a Kaggle Notebook environment
DATA_DIR = "/kaggle/input/datasets/purvanshjoshi1/healthcare" if os.path.exists("/kaggle/input/") else "./"  
# Note: Update this to a subfolder like /Training/ if the Kaggle dataset is structured with subfolders.

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
IMG_SIZE = 224
SEED = 42
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Device configuration (Agnostic to CUDA, MPS for Macs, or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================================== #
# PART 1: PREPROCESSING AND DATA CLEANING
# Addressing real-world clinical challenges: inter-patient variability, class imbalance.
# ============================================================================== #
print("\n--- Part 1: Preprocessing & Data Cleaning ---")

# Defining robust augmentations to handle variability and limited annotations
# We use standard ImageNet Normalization 
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), # addressing alignment variabilities
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # addressing scanner variabilities
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset to handle split transforms easily
class BrainTumorDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Loading data paths and labels
all_files = []
all_labels = []

for idx, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_dir):
        files = glob.glob(os.path.join(class_dir, "*.*")) # Get all images
        # Simple data cleaning: valid extensions
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        all_files.extend(files)
        all_labels.extend([idx] * len(files))

print(f"Total dataset size: {len(all_files)}")

# Calculating class weights to handle Class Imbalance
class_counts = Counter(all_labels)
print(f"Class distribution: {dict(zip(CLASSES, [class_counts[i] for i in range(len(CLASSES))]))}")
total_samples = len(all_labels)
class_weights = [total_samples / class_counts[i] for i in range(len(CLASSES))]
weights_tensor = torch.FloatTensor(class_weights).to(device)

# Stratified Splitting for structured validation (Train: 80%, Val: 20%)
X_train, X_val, y_train, y_val = train_test_split(all_files, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels)

train_dataset = BrainTumorDataset(X_train, y_train, transform=train_transforms)
val_dataset = BrainTumorDataset(X_val, y_val, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================================== #
# PART 2: IMPLEMENT THE DETECTION ALGORITHM
# Designing an architecture to improve generalization and reliability.
# ============================================================================== #
print("\n--- Part 2: Model Architecture & Detection Algorithm ---")

# Using Pretrained ResNet50 as a robust feature extractor and adapting its head
class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorClassifier, self).__init__()
        # Load a high-capability pretrained model
        weights = models.ResNet50_Weights.DEFAULT
        self.base_model = models.resnet50(weights=weights)
        
        # Replace the fully connected layer 
        num_ftrs = self.base_model.fc.in_features
        # Adding dropout for regularization
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

model = BrainTumorClassifier(num_classes=len(CLASSES)).to(device)
print(f"Model initialized. Expected output classes: {len(CLASSES)}")

# Objective (Loss) and optimizer.
# We incorporate the computed class weights into the loss function to penalize majority classes less
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ============================================================================== #
# PART 3 & 4: ACCURACY, EVALUATION METRIC, F1 SCORE & TRAINING LOOP
# ============================================================================== #
print("\n--- Part 3 & 4: Training & Comprehensive Evaluation ---")

best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    # Training Loop
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    
    # Evaluation Loop
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    val_loss = val_loss / len(val_dataset)
    
    # -- Applying the requested evaluation metrics --
    acc = accuracy_score(all_targets, all_preds)
    # Using 'macro' F1-score to treat all classes equally regardless of support
    f1 = f1_score(all_targets, all_preds, average='macro') 
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro')
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"--> Val Accuracy: {acc:.4f} | Validation F1 Score: {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
    
    # Save best model
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_tumor_model.pth")
        print("    [!] New best model saved based on F1 Score.")

print("\n--- Training Completed ---")

# ============================================================================== #
# FINAL EVALUATION: CONFUSION MATRIX
# ============================================================================== #
if os.path.exists("best_tumor_model.pth"):
    model.load_state_dict(torch.load("best_tumor_model.pth", map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Plotting Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix of Best Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved testing confusion matrix to confusion_matrix.png")
