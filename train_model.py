import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from collections import Counter
from PIL import Image

# Quick Setup
DATA_DIR = "/kaggle/input/datasets/purvanshjoshi1/healthcare" if os.path.exists("/kaggle/input/") else "./"  
BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_SIZE, SEED = 32, 20, 1e-4, 224, 42
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Use GPU if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Working on {device}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# Prepare Data
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
        self.file_paths, self.labels, self.transform = file_paths, labels, transform
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

# Load and clean paths
all_files, all_labels = [], []
for idx, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_dir):
        files = [f for f in glob.glob(os.path.join(class_dir, "*.*")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_files.extend(files)
        all_labels.extend([idx] * len(files))

# Balanced weights to handle class imbalance
class_counts = Counter(all_labels)
weights_tensor = torch.FloatTensor([len(all_labels)/class_counts[i] for i in range(len(CLASSES))]).to(device)

X_train, X_val, y_train, y_val = train_test_split(all_files, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels)

train_loader = DataLoader(BrainTumorDataset(X_train, y_train, train_transforms), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(BrainTumorDataset(X_val, y_val, val_transforms), batch_size=BATCH_SIZE, shuffle=False)

# Model: ResNet50 modified for 4-class classification
class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorClassifier, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Separate the backbone into stages for multi-scale access
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Multi-Scale Fusion Head: Fuses 1024 (Layer 3) + 2048 (Layer 4)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 2048, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(), # Advanced Swish activation
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        l4 = self.layer4(l3)
        
        # Dual-scale Global Average Pooling
        f3 = torch.mean(l3, dim=[2, 3])
        f4 = torch.mean(l4, dim=[2, 3])
        
        return self.classifier(torch.cat([f3, f4], dim=1))

model = BrainTumorClassifier(num_classes=len(CLASSES)).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training Loop
print("\nStarting training loop...")
best_f1 = 0.0
save_path = "/kaggle/working/best_tumor_model.pth" if os.path.exists("/kaggle/working/") else "best_tumor_model.pth"

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    # Evaluate
    model.eval()
    val_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    # Calculate performance metrics
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    scheduler.step(val_loss / len(val_dataset))
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_dataset):.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), save_path)
        print(f" [!] Better model found. Weights saved.")

# Visualize results
if os.path.exists(save_path): model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs.to(device))
        y_true.extend(labels.numpy())
        y_pred.extend(outputs.argmax(1).cpu().numpy())

# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Model Confusion Matrix")
plt.savefig("confusion_matrix.png")

# Explainability with Grad-CAM
import cv2
class GradCam:
    def __init__(self, model, target_layer):
        self.model, self.target_layer = model, target_layer
        self.gradients, self.activations = None, None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    def save_activation(self, m, i, o): self.activations = o
    def save_gradient(self, m, gi, go): self.gradients = go[0]
    def __call__(self, x):
        self.model.zero_grad(); out = self.model(x)
        idx = out.argmax(dim=1).item(); out[0, idx].backward()
        grads = self.gradients[0].cpu().data.numpy(); acts = self.activations[0].cpu().data.numpy()
        weights = np.mean(grads, axis=(1, 2))
        cam = np.maximum(np.sum(acts * weights[:, None, None], axis=0), 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        return cam / (cam.max() + 1e-7), idx

# Showcase Grad-CAM on a random sample from the validation batch
samples, labels = next(iter(val_loader))

# Let's pick a random index from the batch to see different tumors each time
idx = np.random.randint(0, len(samples))
sample_input = samples[idx:idx+1].to(device)
true_label_idx = labels[idx].item()

# Generate the Heatmap (Targeting the final spatial layer)
cam_gen = GradCam(model, model.layer4[-1])
cam, pred_idx = cam_gen(sample_input)

# De-normalize and prepare image for display
img = samples[idx].permute(1,2,0).numpy()
img = np.clip(np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406]), 0, 1)

# Apply Colormap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1] / 255.0

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(img); plt.title(f"True Label: {CLASSES[true_label_idx]}"); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(heatmap * 0.4 + img * 0.6); plt.title(f"AI Attention (Pred: {CLASSES[pred_idx]})"); plt.axis('off')
plt.savefig("explainability.png")
print(f"Showing Grad-CAM for a random {CLASSES[true_label_idx]} samples.")

# Final Scores
print(f"\nFinal Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Final F1-Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
