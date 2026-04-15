# Neural-NEXUS: Brain Tumor Analysis

This repository houses a comprehensive, AI/ML-driven deep learning system designed for robust brain tumor analysis and classification using MRI imaging. The system was meticulously developed to combat fundamental real-world clinical challenges such as dataset class disparity, inter-patient variability, and black-box interpretability issues.

##  Key Features

- **Automated Kaggle Path Discovery:** Dynamically locates and normalizes tumor classes regardless of Kaggle directory structures to prevent execution failures.
- **Advanced Preprocessing:** Employs ImageNet stabilization and aggressive geometry/color augmentations (rotations, flips, jitters) to drastically enhance clinical generalization.
- **Dynamic Inverse-Class Weighting:** Computationally resolves extreme dataset imbalances, ensuring rare classes are mathematically penalized equal to common classes.
- **Deep Transfer Learning:** Replaces primitive architectures with a robust `EfficientNet/ResNet50` classifier backbone customized with high dropout (0.5) to isolate pure tumor features.
- **Clinical Explainability (Grad-CAM):** Instead of issuing blind predictions, the architecture overlays vivid heatmaps onto the native MRI structure, highlighting the specific structural anomalies the AI evaluated to form its diagnosis.

---

## Confusion Matrix: Clinical Performance Analysis
<img width="1392" height="1046" alt="image" src="https://github.com/user-attachments/assets/3f6a8497-37ef-4abd-9422-00cd68ef4604" />

## Model Predictions:

## Predisction 2
<img width="1013" height="492" alt="image" src="https://github.com/user-attachments/assets/6d22032e-b411-4c6c-a83d-e0a391909178" />

## Prediction 2
<img width="895" height="461" alt="Screenshot 2026-04-15 at 5 10 05 PM" src="https://github.com/user-attachments/assets/f49be335-3013-437c-a29b-6b7121c7c2f4" />
The model is identifying no tumour by looking at the specific healthy part of the brain.


