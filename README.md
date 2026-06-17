# 🫁 ChestMamba: Vision Mamba-Based Framework for Pneumonia X-ray Generation and Classification

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-orange)

---

## 📌 Overview

**ChestMamba** is a unified deep learning framework for **pneumonia and COVID-19 chest X-ray analysis**, integrating:

- 🧠 Vision Mamba (VMamba) backbone
- 🎨 VMamba-GAN for synthetic medical image generation
- ⚙️ WGAN-style stable adversarial training
- 📊 Classification with improved VMamba + GAM optimization
- 🧪 CLAHE-based medical image enhancement

The framework jointly improves:
- Data scarcity problems
- Class imbalance issues
- Robustness of pneumonia classification

---

## 🚀 Key Contributions

- 🔥 First integration of **Vision Mamba into GAN (VMamba-GAN)**
- 🎨 High-quality chest X-ray synthesis using VSS Blocks
- ⚖️ WGAN-style training with weight clipping stability
- 📈 Significant improvement in classification accuracy (95.41%)
- 🧠 Dual-stage pipeline: Generation → Classification
- 📊 Strong generalization across multi-class pneumonia datasets

---

## 🧠 Method Overview

### 🔷 Overall Pipeline

```
Input X-ray Images
        │
        ▼
   CLAHE Enhancement
        │
        ▼
   VMamba-GAN Training
        │
        ├──► Synthetic X-ray Generation
        │
        ▼
Augmented Dataset (Real + Fake)
        │
        ▼
   VMamba Classifier + GAM
        │
        ▼
   Final Diagnosis Output
```

---

## 🏗️ Model Architecture

### 🎨 VMamba-GAN
- Generator: VSS Block + Patch Expanding
- Discriminator: VSS Block + Patch Merging
- Loss: Wasserstein loss
- Stabilization:
  - Weight clipping (±0.01)
  - AdamW optimizer
  - Cosine LR scheduling (optional)

### 🧠 VMamba Classifier
- Vision State Space Model backbone
- GAM optimizer for improved generalization
- Pretrained initialization

---

## 📊 Experimental Results

### 🧪 Classification Performance

| Model | Accuracy |
|------|---------|
| VGG16 | 91.62% |
| ResNet50 | 91.88% |
| Swin Transformer | 86.21% |
| ViT | 75.41% |
| **ChestMamba (Ours)** | **95.41%** |

---

### 🎨 Image Generation Quality (FID ↓)

| Model | Quality |
|------|--------|
| WGAN | Poor |
| DCGAN | Poor |
| SAGAN | Medium |
| ViTGAN | Good |
| **VMamba-GAN (Ours)** | **Best** |

---

## ⚙️ Installation

```bash
git clone https://github.com/QiumeiPuAIGroup/ChestMamba.git
cd ChestMamba

pip install -r requirements.txt
```

---

## 🏋️ Training

Three classes
```bash
python train_3_mymodel.py \
  --n_epochs 50 \
  --batch_size 8 \
  --img_size 256 \
  --latent_dim 100 \
  --model-name mamba_gan \
  --data-path data/COVID-19_Radiography_Dataset/COVID/images \
  --save-path checkpoints_gan/GAN/MambaGan_myself
```

Four classes
```bash
python train_4_classify_mymodel.py \
  --n_epochs 50 \
  --batch_size 8 \
  --img_size 256 \
  --latent_dim 100 \
  --model-name mamba_gan \
  --data-path data/COVID-19_Radiography_Dataset/COVID/images \
  --save-path checkpoints_gan/GAN/MambaGan_myself
```

Using GAM's four-class classification
```bash
python train_4_classify_mymodel_gam.py \
  --n_epochs 50 \
  --batch_size 8 \
  --img_size 256 \
  --latent_dim 100 \
  --model-name mamba_gan \
  --data-path data/COVID-19_Radiography_Dataset/COVID/images \
  --save-path checkpoints_gan/GAN/MambaGan_myself
```

---

## 🧪 Evaluation Pipeline

### 1️⃣ Train GAN and Generate Images

```bash
python train_4_mygan.py
```

Using bceloss
```bash
python train_4_mygan_bceloss.py
```

### 2️⃣ Outputs
- fake images → `fake_imgs/`
- test images → `test_imgs/`
- checkpoints → `.pth`

---

## 🔬 Ablation Study Summary

| Component | Effect |
|----------|--------|
| CLAHE | + contrast enhancement |
| VMamba-GAN | + data diversity |
| Pretraining | + convergence |
| GAM | + robustness |
| Full Model | best performance |

---

## 📁 Dataset

- COVID-19 Radiography Dataset  
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

- Chest X-ray Dataset  
https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

---

## 📬 Contact

📧 puqiumei@muc.edu.cn

---

## ⚠️ License

Academic research use only.
