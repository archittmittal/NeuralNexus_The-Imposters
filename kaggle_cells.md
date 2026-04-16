# Brain Tumor Analysis: ResNet50 + Grad-CAM
Copy these cells into your Kaggle environment for a robust, explainable MRI classification model.

---

### **Cell 1: Setup & Discovery**
```python
import os, glob, torch, torch.nn as nn, torch.optim as optim, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter
from PIL import Image

# Config
BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_SIZE, SEED = 32, 20, 1e-4, 224, 42
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
base_dir = "/kaggle/input" if os.path.exists("/kaggle/input") else "./"

# Automatically find any images and categorize them by folder name
all_images = [f for f in glob.glob(os.path.join(base_dir, "**", "*.*"), recursive=True) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
CLASSES = sorted(list(set(os.path.basename(os.path.dirname(f)) for f in all_images if os.path.basename(os.path.dirname(f)).lower() not in ['train', 'test', 'val'])))
class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

files, labels = [], []
for f in all_images:
    cls = os.path.basename(os.path.dirname(f))
    if cls in class_to_idx:
        files.append(f); labels.append(class_to_idx[cls])

print(f"Discovered {len(files)} images across {len(CLASSES)} classes on {device}")
```

---

### **Cell 2: Preprocessing**
```python
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class TumorDataset(Dataset):
    def __init__(self, fps, lbs, tf=None): self.fps, self.lbs, self.tf = fps, lbs, tf
    def __len__(self): return len(self.fps)
    def __getitem__(self, i):
        img = Image.open(self.fps[i]).convert("RGB")
        if self.tf: img = self.tf(img)
        return img, self.lbs[i]

# Compute class weights for better balance
counts = Counter(labels)
weights_tensor = torch.FloatTensor([len(labels)/counts[i] for i in range(len(CLASSES))]).to(device)

X_train, X_val, y_train, y_val = train_test_split(files, labels, test_size=0.15, random_state=SEED, stratify=labels)
train_loader = DataLoader(TumorDataset(X_train, y_train, train_tf), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TumorDataset(X_val, y_val, val_tf), batch_size=BATCH_SIZE, shuffle=False)
```

---

### **Cell 3: Advanced Architectures**
*Choose one of the models below. The Attentional version is recommended for maximum precision.*

```python
# --- Option A: Scale-Refined ResNet ---
class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorClassifier, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.classifier = nn.Sequential(nn.Linear(1024 + 2048, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(0.4), nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        l3 = self.layer3(x); l4 = self.layer4(l3)
        return self.classifier(torch.cat([torch.mean(l3, dim=[2, 3]), torch.mean(l4, dim=[2, 3])], dim=1))

# --- Option B: Attentional Multi-Scale (SOTA) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_planes, in_planes // ratio), nn.ReLU(), nn.Linear(in_planes // ratio, in_planes))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        return self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)

class AttentionalBrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(AttentionalBrainTumorClassifier, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.ca3 = ChannelAttention(1024)
        self.ca4 = ChannelAttention(2048)
        self.classifier = nn.Sequential(nn.Linear(1024 + 2048, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(0.4), nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        l3 = self.layer3(x); l3 = l3 * self.ca3(l3)
        l4 = self.layer4(l3); l4 = l4 * self.ca4(l4)
        return self.classifier(torch.cat([torch.mean(l3, dim=[2, 3]), torch.mean(l4, dim=[2, 3])], dim=1))

# Initialize (Using Attentional version as requested/used by user)
model = AttentionalBrainTumorClassifier(num_classes=len(CLASSES)).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
```

---

### **Cell 4: Training Loop**
```python
best_f1, save_path = 0.0, "/kaggle/working/best_tumor_model.pth"

for epoch in range(EPOCHS):
    model.train(); r_loss = 0.0
    for inputs, lbs in train_loader:
        inputs, lbs = inputs.to(device), lbs.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), lbs)
        loss.backward(); optimizer.step()
        r_loss += loss.item() * inputs.size(0)
    
    model.eval(); v_loss, all_p, all_t = 0.0, [], []
    with torch.no_grad():
        for inputs, lbs in val_loader:
            inputs = inputs.to(device)
            out = model(inputs)
            v_loss += criterion(out, lbs.to(device)).item() * inputs.size(0)
            all_p.extend(out.argmax(1).cpu().numpy()); all_t.extend(lbs.numpy())
            
    f1 = f1_score(all_t, all_p, average='macro')
    scheduler.step(v_loss / len(y_val))
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {r_loss/len(X_train):.4f} | F1: {f1:.4f} | Acc: {accuracy_score(all_t, all_p):.4f}")
    
    if f1 > best_f1:
        best_f1 = f1; torch.save(model.state_dict(), save_path); print(" [!] Weights saved.")
```

---

### **Cell 5: Confusion Matrix**
```python
if os.path.exists(save_path): model.load_state_dict(torch.load(save_path, map_location=device))
model.eval(); all_p, all_t = [], []

with torch.no_grad():
    for inputs, lbs in val_loader:
        all_p.extend(model(inputs.to(device)).argmax(1).cpu().numpy()); all_t.extend(lbs.numpy())

plt.figure(figsize=(8,6)); sns.heatmap(confusion_matrix(all_t, all_p), annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Model Confusion Matrix"); plt.show()
```

---

### **Cell 6: Clinical Interpretability (Grad-CAM)**
```python
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
        return cv2.resize(cam, (x.shape[3], x.shape[2])) / (cam.max() + 1e-7), idx

# Run on a random sample from the validation batch
inputs, labels = next(iter(val_loader))
idx = np.random.randint(0, len(inputs)) # Pick a different patient every time
sample_input = inputs[idx:idx+1].to(device)

cam_gen = GradCam(model, model.layer4[-1])
cam, pred = cam_gen(sample_input)

# De-normalize and prepare for display
img = np.clip(np.array([0.229, 0.224, 0.225]) * inputs[idx].permute(1,2,0).numpy() + np.array([0.485, 0.456, 0.406]), 0, 1)
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1] / 255.0

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(img); plt.title(f"True: {CLASSES[labels[idx]]}"); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(heatmap * 0.4 + img * 0.6); plt.title(f"AI Attention (Pred: {CLASSES[pred]})"); plt.axis('off')
plt.show()
```

---

### **Cell 7: Final Metrics Summary**
```python
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval(); all_p, all_t = [], []
with torch.no_grad():
    for inputs, lbs in val_loader:
        all_p.extend(model(inputs.to(device)).argmax(1).cpu().numpy()); all_t.extend(lbs.numpy())

print(f"Final Accuracy: {accuracy_score(all_t, all_p):.4f}")
print(f"Final F1-Score: {f1_score(all_t, all_p, average='macro'):.4f}")
```
