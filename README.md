# Image Classification with Localization and Augmentation (PyTorch & TensorFlow)

This repository contains implementations of object classification with localization (bounding box prediction) using both PyTorch and TensorFlow. The models have been trained on the Caltech-101 dataset with varying subsets and methodologies.

## Update
- **New file added:** `classification-with-localization-pytorch-with-augmentation.ipynb`
  - Contains a PyTorch model trained on the full Caltech-101 dataset.
  - Improved performance using data augmentation with the **Albumentations** library.

---

## PyTorch: Classification with Localization (with Augmentation)

In this project, I implemented object classification with localization using the **Caltech-101** dataset.

### Model Details:
- **Base Model:** Pre-trained `mobilenet_v3_large` with two output heads:
  - One for predicting the **class label**.
  - One for predicting **bounding box coordinates**.

### Results:
- **Training Class Accuracy:** 97.46%
- **Training Bounding Box Loss:** 0.0035
- **Validation Accuracy:** 86.31%
- **Validation Bounding Box Loss:** 0.0073
- **Test Accuracy:** 88.89%
- **Test Bounding Box Loss:** 0.0067

> **Note:** The bounding box loss is significantly smaller compared to the TensorFlow version due to normalization of the bounding box coordinates.

---
![__results___32_0](https://github.com/user-attachments/assets/994bf7c0-8dbf-42b4-b80c-bab95ec7c8eb)
---

## TensorFlow: Classification with Localization (Without Augmentation)

In this project, I implemented classification with localization using a portion of the **Caltech-101** dataset, specifically the following categories: `['airplanes', 'Faces', 'Motorbikes']`.

### Model Versions:
1. **Base Model:** `NasNet`
   - **Training Class Accuracy:** 100%
   - **Training Bounding Box Loss:** 1551.15
   - **Validation Accuracy:** 100%
   - **Validation Bounding Box Loss:** 1293.05

2. **Base Model:** `VGG-16`
   - **Training Class Accuracy:** 99.84%
   - **Training Bounding Box Loss:** 688.58
   - **Validation Accuracy:** 100%
   - **Validation Bounding Box Loss:** 802.32

### Additional Notes:
- All custom functions for data parsing and image visualization are included in `my_helper_funcs.py`.
- **Version 3:** Check out this [Google Drive link](https://drive.google.com/drive/folders/1I55CnyZbGj2Gsg5IVqeIaKitRl8WpA97?usp=drive_link) for more details.
  
> **Note:** Image augmentation has not been implemented in this version.

---

Feel free to clone the repository, explore the notebooks, and contribute!
