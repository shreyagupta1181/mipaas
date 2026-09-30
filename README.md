# MIPaaS — Chest X-ray Multi-Label Disease Classification

MIPaaS is a deep-learning based medical image classification project for detecting multiple thoracic diseases from chest X-ray images.

The system performs **multi-label classification**, meaning a single chest X-ray can be associated with multiple diseases simultaneously.

> **Research use only:** This project is intended for research, experimentation, and educational purposes. It is not a clinically validated diagnostic system and should not be used as a substitute for professional medical diagnosis.

---

## Overview

The project uses the **NIH ChestX-ray14** dataset and investigates multiple deep-learning approaches for 14 thoracic disease labels.

### Disease Classes

The model predicts the following 14 conditions:

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural Thickening
13. Pneumonia
14. Pneumothorax

Because this is a multi-label problem, multiple classes can be predicted for the same X-ray.

---

## Project Pipeline

```text
Chest X-ray
     │
     ▼
Grayscale conversion
     │
     ▼
224 × 224 preprocessing
     │
     ▼
CNN backbone
     │
     ├── ResNet-18
     │
     └── DenseNet-121
     │
     ▼
14 output logits
     │
     ▼
Sigmoid probabilities
     │
     ▼
Per-disease thresholds
     │
     ▼
Predicted disease labels