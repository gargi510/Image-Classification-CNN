# Ninjacart Vegetable Image Classifier

A multi-class image classification system to identify vegetables — **potato**, **onion**, **tomato**, and **indian market (noise)** — built for Ninjacart's supply chain automation pipeline.

---

## Dataset

- 3,135 training images | 351 test images | 4 classes
- Source: [Google Drive]

| Class | Train | Test |
|---|---|---|
| Potato | 898 | 81 |
| Onion | 849 | 83 |
| Tomato | 789 | 106 |
| Indian Market | 599 | 81 |

---

## Pipeline

- **EDA** — class distribution, image dimension analysis, aspect ratio visualisation
- **Preprocessing** — square center-crop → 128×128 resize → normalisation
- **Augmentation** — random flip, rotation, zoom (training only)
- **Models** — Baseline CNN → Improved CNN → VGG16 / ResNet50 / MobileNetV2 (fine-tuned) → Hyperparameter Tuning
- **Evaluation** — accuracy, classification report, confusion matrix, random sample predictions

---

## Results

| Model | Test Accuracy |
|---|---|
| Baseline CNN | 73.79% |
| Improved CNN | 85.19% |
| MobileNetV2 (fine-tuned) | 91.45% |
| ResNet50 (fine-tuned) | 92.02% |
| VGG16 (fine-tuned) | 93.45% |
| **VGG16 (HP tuned)** | **96.87%** |

---

## Tech Stack

`TensorFlow` `Keras` `Keras Tuner` `NumPy` `Pandas` `Matplotlib` `Seaborn` `Scikit-learn`

---

## Usage

1. Open `image_classification.ipynb` in Google Colab
2. Run all cells sequentially
3. Dataset downloads automatically via `gdown`

---

## Key Findings

- Improving the scratch CNN (BatchNorm + L2 + Dropout) gave the largest single gain: **+11.4%**
- Transfer learning outperformed scratch CNNs by **~8%** with no additional data
- Hyperparameter tuning pushed VGG16 from 93.45% to **96.87%**
- Tomato classified perfectly (F1 = 1.00); onion–potato confusion was the primary error mode
- MobileNetV2 (9.87 MB vs VGG16's 56.64 MB) is recommended for edge deployment
