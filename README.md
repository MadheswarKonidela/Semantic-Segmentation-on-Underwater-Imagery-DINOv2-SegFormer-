# Semantic-Segmentation-on-Underwater-Imagery-DINOv2-SegFormer-
A professional and well-organized `README.md` is crucial for making your project reproducible. Below is a structured template that highlights your sophisticated approach (DINOv2 + SegFormer) and provides clear instructions for others.

---

# Underwater Semantic Segmentation: DINOv2 + SegFormer

This repository implements a high-performance semantic segmentation pipeline for underwater environments using the **SUIM (Semantic Underwater Image Segmentation)** dataset. We leverage the powerful self-supervised features of **DINOv2** and the efficient multi-scale decoding of **SegFormer**.

---

## Prerequisites & Environment

* **Python Version:** `3.9` or higher
* **Core Libraries:**
* `torch==2.1.x`
* `torchvision==0.16.x`
* `timm==0.9.x`
* `albumentations==1.3.x`
* `opencv-python==4.8.x`
* `PyYAML`
* `tqdm`



### Quick Install

```bash
pip install torch torchvision timm albumentations opencv-python PyYAML tqdm

```

---

##  How to Run

### 1. Data Preparation

Organize your SUIM dataset in the project root as follows:

```text
Dataset/
└── train_val/
    ├── images/  # (.jpg files)
    └── masks/   # (.bmp color-coded files)

```

### 2. Training

Adjust your hyperparameters in `configs/segformer_head_config.yaml`, `configs/unet_head_config.yaml`  and run:

```bash
python3 -m src.train --config configs/segformer_head_config.yaml

```
```bash
python3 -m src.evalute --config configs/segformer_head_config.yaml

```
```bash
python3 -m src.train --config configs/unet_head_config.yaml

```

### 3. Visual Verification

Use the provided Jupyter Notebook to verify that the RGB masks are correctly remapped to class IDs:

```bash
jupyter notebook notebooks/verify_masks.ipynb

```

---

## Project Approach

### 1. Model Architecture

* **Backbone:** **DINOv2 (ViT-L/14)**. We utilize a Vision Transformer pre-trained with self-supervision on 142M images, providing robust features that handle underwater distortions (blur, color cast) better than standard CNNs.
* **Segmentation Head:** **SegFormer MLP Decoder**. This head aggregates multi-scale features from intermediate transformer blocks to produce high-resolution segmentation maps.

### 2. Data Strategy

* **RGB-to-ID Mapping:** Underwater masks in SUIM are RGB encoded (e.g., `[255, 255, 0]` for Fish). We implemented a custom remapping layer to convert these to integer IDs (0-7) for cross-entropy calculation.
* **Augmentation:** We used the **Albumentations** library to simulate varying underwater conditions using Random Brightness/Contrast, Hue/Saturation shifts, and Gaussian Noise.

### 3. Optimization

* **Optimizer:** **AdamW** with decoupled weight decay to prevent overfitting.
* **Scheduler:** **Cosine Annealing** for a smooth learning rate decay.
* **Fine-tuning:** We implemented **Progressive Unfreezing**, keeping the early backbone layers frozen while fine-tuning the later transformer blocks to adapt to underwater-specific textures.

---

## Results

| Metric | Value |
| --- | --- |
| **mIoU** | **72.9%** |
| **Global Pixel Accuracy** | **~91.4%** |
| **Training Epochs** | 100 

> **Note:** The current performance is a baseline. We observed that the model began to plateau around epoch 60. Future iterations will include **Linear Warmup** and **Dice Loss** to better address class imbalances in rare marine objects.

---

## Repository Structure

* `src/data/`: Dataset loaders and mask remapping logic.
* `src/models/`: DINOv2 backbone and SegFormer and Unet head definitions.
* `configs/`: YAML files for experiment management.
* `src/train.py` and `src/evaluate.py`: Main training and validation script.

