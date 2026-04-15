# Kaggle Notebook Structure

Copy and paste the following Python code blocks into separate cells in your Kaggle Notebook. 
Set the Kaggle environment to **GPU T4x2** or **P100** before running for fast execution!

---

### **Cell 1: Setup & Imports**
```python
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
# Configuration
# ============================================================================== #
DATA_DIR = "/kaggle/input/datasets/purvanshjoshi1/healthcare" 
# Note: Add '/Training/' to the path above if the tumor folders are inside a Training folder.

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
IMG_SIZE = 224
SEED = 42
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Use Kaggle GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(SEED)
np.random.seed(SEED)
```

---

### **Cell 2: Preprocessing and Data Cleaning**
*(Handles dataset parsing, splitting, data augmentation, and computing Custom Class Weights for Imbalance)*

```python
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), 
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

all_files = []
all_labels = []

# Data Parsing
for idx, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_dir):
        files = glob.glob(os.path.join(class_dir, "*.*"))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        all_files.extend(files)
        all_labels.extend([idx] * len(files))

print(f"Total dataset size: {len(all_files)}")

if len(all_files) == 0:
    raise ValueError(f"CRITICAL ERROR: No images found! The DATA_DIR '{DATA_DIR}' might be incorrect.\nIf your Kaggle dataset has a folder like 'Training', change Cell 1's path to: DATA_DIR = '{DATA_DIR}/Training'")

# Compute class weights for imbalance safely
class_counts = Counter(all_labels)
print(f"Class distribution: {dict(zip(CLASSES, [class_counts.get(i, 0) for i in range(len(CLASSES))]))}")

class_weights = []
for i in range(len(CLASSES)):
    count = class_counts.get(i, 0)
    # Prevent Division By Zero
    weight = len(all_labels) / count if count > 0 else 1.0
    class_weights.append(weight)

weights_tensor = torch.FloatTensor(class_weights).to(device)

# 80/20 train/validation split
X_train, X_val, y_train, y_val = train_test_split(all_files, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels)

train_dataset = BrainTumorDataset(X_train, y_train, transform=train_transforms)
val_dataset = BrainTumorDataset(X_val, y_val, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
```

---

### **Cell 3: Implement the Detection Algorithm**
*(Sets up the neural network architecture using Transfer Learning)*

```python
class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorClassifier, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.base_model = models.resnet50(weights=weights)
        
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5), # Regularization
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

model = BrainTumorClassifier(num_classes=len(CLASSES)).to(device)
print("Model initialized!")

# Setup Loss (using weighted tensor from previous cell) + Optimizer
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
```

---

### **Cell 4: Accuracy, F1 Score & Training Loop**
*(Runs the training process and evaluates Accuracy, Precision, Recall, and the Macro F1-Score at every epoch)*

```python
best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    # Train
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    
    # Evaluate
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    val_loss = val_loss / len(val_dataset)
    
    # Metrics calculation
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro') 
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro')
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"--> Accuracy: {acc:.4f} | F1 Score: {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
    
    # Save the highest F1-Score model
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "/kaggle/working/best_tumor_model.pth")

print("Training cycle completed.")
```

---

### **Cell 5: Evaluating with Confusion Matrix**
*(Generates the visual analysis of accuracy across the different classes)*

```python
# Load the best weights
if os.path.exists("/kaggle/working/best_tumor_model.pth"):
    model.load_state_dict(torch.load("/kaggle/working/best_tumor_model.pth", map_location=device))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Display Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix (Best Model)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()  # Display directly in Kaggle Notebook
```
