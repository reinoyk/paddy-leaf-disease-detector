# PADDY-LEAF-DISEASE-DETECTOR

*Detects Crop Diseases, Protects Your Harvest Instantly*

<p align="center">
<img src="https://img.shields.io/github/last-commit/reinoyk/paddy-leaf-disease-detector?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/reinoyk/paddy-leaf-disease-detector?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/reinoyk/paddy-leaf-disease-detector?style=flat&color=0080ff" alt="repo-language-count">
</p>

<em>Built with Python, PyTorch, and Jupyter Notebook</em>

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Why This Project?](#why-this-project)
* [Model](#model)
* [Dataset](#dataset)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Testing](#testing)
* [Results](#results)
* [How to Contribute](#how-to-contribute)
* [License](#license)

---

## Overview

**paddy-leaf-disease-detector** is a machine learning project designed to identify and classify diseases in paddy leaves using lightweight deep learning models. The project is optimized for rapid inference, making it suitable for real-world deployment in resource-constrained environments such as edge devices or mobile applications. By providing accurate disease identification, this tool aims to help farmers make timely interventions, reduce crop losses, and support sustainable agricultural practices.

---

## Features

* 🚀 **Optimized Model:** Utilizes MobileNetV2 for fast and resource-efficient inference.
* 📦 **Pre-trained Weights:** Includes a ready-to-use `best_mobilenetv2.pth` model for instant deployment or further fine-tuning.
* 📊 **Reproducible Training:** Comes with a complete Jupyter Notebook containing training, validation, and evaluation workflows.
* 🌱 **Agriculture Focused:** Specially tailored to detect major paddy leaf diseases using real-world field images.
* ⚙️ **Easy Integration:** Designed for seamless adoption into existing research, farm monitoring systems, or mobile apps.
* 📈 **Robust Evaluation:** Supports evaluation metrics such as accuracy, precision, recall, and F1-score for comprehensive model assessment.

---

## Why This Project?

The early and accurate detection of paddy leaf diseases is crucial for maximizing crop yield and minimizing financial loss for farmers. Manual identification is time-consuming and requires expert knowledge, which may not always be available. By leveraging deep learning, this project automates the process, enabling rapid and reliable diagnosis accessible to everyone.

---

## Model

The primary model implemented in this project is **MobileNetV2**, a state-of-the-art lightweight convolutional neural network (CNN) designed for high performance and efficiency. MobileNetV2 is particularly well-suited for mobile and embedded applications due to its small memory footprint and fast inference speed. For benchmarking, the project can be extended to compare other models such as EfficientNet-B0.

**Key Model Details:**

* Architecture: MobileNetV2
* Input: RGB images (224x224)
* Output: Multi-class classification (10 paddy leaf disease categories)
* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Metrics: Accuracy, Precision, Recall, F1-Score

---

## Dataset
https://www.kaggle.com/datasets/imbikramsaha/paddy-doctor

---

## Getting Started

### Prerequisites

* Python 3.8+
* Jupyter Notebook
* PyTorch >= 1.12
* torchvision
* numpy, pandas, scikit-learn, matplotlib

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/reinoyk/paddy-leaf-disease-detector
   ```

2. **Navigate to the project directory:**

   ```sh
   cd paddy-leaf-disease-detector
   ```
   
3. **Download the dataset**
  
4. **Install the dependencies:**


### Usage

* Open the Jupyter notebook (`classification.ipynb`).
* Follow the steps to run data preprocessing, training, validation, and testing.
* To use the trained model for prediction:

  ```python
  # Example usage
  from model import load_model, predict
  model = load_model('best_mobilenetv2.pth')
  result = predict(model, 'path/to/leaf_image.jpg')
  print(result)
  ```

### Testing

* The notebook includes test code for model evaluation.
* You can test the model on new images by providing the image path in the notebook or via a Python script.

---

## Results

* Achieves high F1-score and accuracy on the test set (see notebook for detailed metrics).
* Example classification report and confusion matrix are provided for performance visualization.
* MobileNetV2 outperforms heavier models in inference time, while maintaining competitive accuracy.

---

## How to Contribute

We welcome contributions! Please open issues or pull requests for bug fixes, new features, or improvements. For major changes, please discuss them first via an issue.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
