# 📸 Deep Learning: Image Classification using CNN and Transfer Learning

This project explores **image classification on the CIFAR-10 dataset** using two approaches:  
1. Building a **Convolutional Neural Network (CNN) from scratch**  
2. Implementing **Transfer Learning with MobileNet**  

The project demonstrates how custom-built CNNs compare to transfer learning models in terms of performance, training time, and generalization ability.

---

## 📌 1. General Description
- Built a CNN model to classify CIFAR-10 images into predefined categories.  
- Implemented a transfer learning approach using the pre-trained **MobileNet** model.  
- Compared both models using evaluation metrics and visualizations.  

---

## 📊 2. Project Overview
### 🔍 Features
- Performs image classification on the CIFAR-10 dataset (60,000 images, 10 classes).  
- Implements **two modeling approaches**: CNN from scratch and MobileNet transfer learning.  
- Evaluates and compares models using **accuracy, loss, and confusion matrices**.  

### 🎯 Problem Solved
- Automates classification of small-scale images into meaningful categories.  
- Demonstrates practical differences between training models from scratch and leveraging pre-trained architectures.  

### 🌍 Potential Impact
- **Education:** Useful for learning CNNs and transfer learning.  
- **Industry:** Shows how transfer learning accelerates training and improves results.  
- **Research:** Provides a foundation for further image classification experiments.  

---

## 📁 3. Dataset Description
- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  
- **Images:** 60,000 (32x32 color images, 10 classes).  
- **Split:** 50,000 training / 10,000 testing.  
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.  
- **Use case:** Benchmark dataset for computer vision tasks.  

---

## 🎯 4. Research Goal
- Classify CIFAR-10 images into 10 categories.  
- Compare CNN vs. MobileNet transfer learning.  
- Evaluate using accuracy, loss, and visualizations.  

---

## ⚙️ 5. Steps Taken
### 1. Data Preprocessing
- Normalized image pixel values.  
- Applied data augmentation to improve generalization.  

### 2. Model Building
- **Custom CNN** with convolutional, pooling, and dense layers.  
- **MobileNet Transfer Learning**, fine-tuning the top layers.  

### 3. Training
- Used Adam optimizer and early stopping.  
- Tuned learning rate for stability.  

### 4. Evaluation
- Compared test accuracy and loss.  
- Visualized training history and confusion matrices.  

---

## 🔍 6. Key Findings
- **MobileNet transfer learning** outperformed the CNN trained from scratch.  
- Transfer learning achieved **higher accuracy with fewer epochs**.  
- The custom CNN worked but required more parameter tuning.  
- MobileNet showed **better generalization** to unseen data.  

---

## 🧪 7. How to Reproduce
### Requirements
- **Python:** 3.11.5  
- **Libraries:**
  ```python
  import tensorflow as tf
  from tensorflow.keras import datasets, layers, models
  import matplotlib.pyplot as plt
  import numpy as np

Files

Project_3_(CNN_LN).ipynb → CNN + MobileNet implementations

Project_3 Presentation.pdf → Project presentation slides

requirements.txt → Dependencies

🚀 8. Next Steps / Improvements

Try other pre-trained models (ResNet, VGG, EfficientNet).

Perform hyperparameter tuning.

Add advanced data augmentation and regularization.

Explore deploying the trained model as a web/app service.

🗂️ 9. Repository Structure
File/Folder	Description
Project_3_(CNN_LN).ipynb	Notebook containing CNN and MobileNet models
Project_3 Presentation.pdf	Slide presentation of project findings
requirements.txt	Python dependencies required to run project
📢 Contact Information

Project Lead: Nekky Lung
Email: nekkytang@gmail.com

LinkedIn: linkedin.com/in/nekkytang

GitHub: https://github.com/nekkylung

Repository: Deep-Learning-Image-Classification-using-CNN-and-Transfer-Learning

📜 License

This project is licensed under the MIT License. See the LICENSE
 file for details.

🙏 Acknowledgments

This project was developed as part of a collaborative team-based analysis, simulating real-world analytics scenarios and demonstrating practical applications of machine learning in computer vision and financial markets.

