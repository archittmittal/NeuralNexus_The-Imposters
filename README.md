# Neural-NEXUS: Brain Tumor Analysis

This repository houses a comprehensive, AI/ML-driven deep learning system designed for robust brain tumor analysis and classification using MRI imaging. The system was meticulously developed to combat fundamental real-world clinical challenges such as dataset class disparity, inter-patient variability, and black-box interpretability issues.

##  Key Features

- **Automated Kaggle Path Discovery:** Dynamically locates and normalizes tumor classes regardless of Kaggle directory structures to prevent execution failures.
- **Advanced Preprocessing:** Employs ImageNet stabilization and aggressive geometry/color augmentations (rotations, flips, jitters) to drastically enhance clinical generalization.
- **Dynamic Inverse-Class Weighting:** Computationally resolves extreme dataset imbalances, ensuring rare classes are mathematically penalized equal to common classes.
- **Deep Transfer Learning:** Replaces primitive architectures with a robust `EfficientNet/ResNet50` classifier backbone customized with high dropout (0.5) to isolate pure tumor features.
- **Clinical Explainability (Grad-CAM):** Instead of issuing blind predictions, the architecture overlays vivid heatmaps onto the native MRI structure, highlighting the specific structural anomalies the AI evaluated to form its diagnosis.


## Dataset Details 
 Kaggle dataset link - https://www.kaggle.com/datasets/purvanshjoshi1/healthcare

##  Model Architecture: ResNet50 for MRI Analysis

For this project, we leveraged **ResNet50**, a deep residual learning framework that is widely considered the "gold standard" for complex image recognition tasks. 

### Why ResNet50?
*   **Deep Feature Extraction**: With 50 layers of depth, the model can "see" subtle textures and density variations in MRI scans that the human eye might miss.
*   **The Residual Advantage**: Its unique "skip-connection" architecture allows it to learn without the risk of losing information in deeper layers (vanishing gradients), making it incredibly stable during training.
*   **Transfer Learning**: We used a model pre-trained on millions of images and fine-tuned it specifically on brain tumor datasets. This gave our AI a massive "head start" in understanding basic shapes and structures before we taught it to find tumors.

---

###  Performance & The Confusion Matrix
Our model achieved a **91.86%(calculated from the confusion matrix as well) overall accuracy**, a strong benchmark for clinical-grade analysis. To understand how "strong" this really is, we look at the **Confusion Matrix**:

1.  **High Diagonal Confidence**: Most predictions fall on the central diagonal line, meaning the model isn't just lucky—it truly knows the difference between a Glioma and a Meningioma.
2.  **Precision Across Classes**: The matrix shows that even with varying sample sizes (class imbalance), the model maintains a high "hit rate" for both common and rare tumor types.
3.  **Trust but Verify**: By seeing where the model *rarely* gets confused, we can identify which tumor types look visually similar (like certain Gliomas vs. Meningiomas), allowing for better human-in-the-loop review.

---

###  Clinical Explainability (Grad-CAM)
We didn't just build a "black box." Our system uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight exactly where the model is looking when it makes a diagnosis.

*   **Heatmap Visualization**: The model generates a thermal map over the MRI. A "hot" red area indicates the specific region the AI identified as suspicious.
*   **Building Trust**: This allows doctors to see *why* the AI predicted a certain tumor type, making the model a collaborative tool rather than just a prediction engine.

---

###  Technical Specs
*   **Optimizer**: AdamW (for better weight decay and generalization)
*   **Scheduler**: ReduceLROnPlateau (automatically slows down learning to "fine-tune" when performance plateaus)
*   **Regularization**: 50% Dropout layer to prevent the model from simply memorizing the training data.


## Confusion Matrix: Clinical Performance Analysis
<img width="1392" height="1046" alt="image" src="https://github.com/user-attachments/assets/3f6a8497-37ef-4abd-9422-00cd68ef4604" />

## Model Predictions:

## Predisction 2
<img width="1013" height="492" alt="image" src="https://github.com/user-attachments/assets/6d22032e-b411-4c6c-a83d-e0a391909178" />
Clinical Heatmap (Grad-CAM)\nTumor Localization

## Prediction 2
<img width="895" height="461" alt="Screenshot 2026-04-15 at 5 10 05 PM" src="https://github.com/user-attachments/assets/f49be335-3013-437c-a29b-6b7121c7c2f4" />
The model is identifying no tumour by looking at the specific healthy part of the brain.

## Prediction 3
<img width="1768" height="926" alt="image" src="https://github.com/user-attachments/assets/d8a946fb-fb62-4c15-8086-2b481171e6fa" />



