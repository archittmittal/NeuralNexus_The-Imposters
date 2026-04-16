import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ============================================================================== #
# EXPLAINABILITY / INTERPRETABILITY: GRAD-CAM
# Leveraging meaningful feature representations for visual explanations.
# ============================================================================== #

class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorClassifier, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Separate the backbone into stages for multi-scale access (matching train_model.py)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 2048, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(), 
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        l3 = self.layer3(x); l4 = self.layer4(l3)
        f3, f4 = torch.mean(l3, dim=[2, 3]), torch.mean(l4, dim=[2, 3])
        return self.classifier(torch.cat([f3, f4], dim=1))

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(x)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward()
        
        # Get gradient and activation maps
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Weight the activations by the gradients (Global Average Pooling on gradients)
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU on CAM
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam, target_class

def show_cam_on_image(img_path, cam, out_path="gradcam_result.png"):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    cam_img = heatmap + img
    cam_img = cam_img / np.max(cam_img)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original MRI")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.uint8(255 * cam_img))
    plt.title("Grad-CAM Interpretability")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Explainability heatmap saved to: {out_path}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = BrainTumorClassifier(num_classes=4).to(device)
    
    weights_path = "best_tumor_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded trained model weights.")
    else:
        print("Warning: Model weights not found. Using untrained model for demonstration.")
    
    # Find a sample image
    sample_img_path = None
    import glob
    candidates = glob.glob("glioma/*.jpg") + glob.glob("meningioma/*.jpg") + glob.glob("notumor/*.jpg") + glob.glob("pituitary/*.jpg")
    
    if candidates:
        sample_img_path = candidates[0]
        print(f"Running Grad-CAM on sample image: {sample_img_path}")
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(sample_img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Specific target layer for ResNet50 (matching the refined architecture)
        target_layer = model.layer4[-1] # Targets the final bottleneck block
        
        # Initialize and run CAM
        cam_generator = GradCam(model, target_layer)
        cam, pred_class = cam_generator(input_tensor)
        
        CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
        print(f"Model Predicted Class Index: {pred_class} -> {CLASSES[pred_class]}")
        
        show_cam_on_image(sample_img_path, cam)
    else:
        print("Please ensure images exist in the class directories to run an explanation.")
