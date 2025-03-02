# Trash Classification

## 📌 Project Overview
This project focuses on classifying waste using **Convolutional Neural Networks (CNNs)**, specifically leveraging **SqueezeNet** and **GoogleNet**. The goal is to develop an efficient deep learning model that accurately classifies different types of trash for automated waste management systems.

## 🏗 Model Architecture
Using two CNN architectures:
- **SqueezeNet**: A lightweight model optimized for speed and efficiency.
- **GoogLeNet (InceptionV1)**: A deep network with inception modules for improved feature extraction.

## 📊 Dataset
The dataset consists of 6 classes:
- **Plastic**
- **Metal**
- **Paper**
- **Glass**
- **Cardboard**
- **Trash**
  
Link dataset: [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification?resource=download)

## 🚀 Steps: 
1. Data Acquisition 
2. Data Preprocessing 
 - Change size to appp model 
 - Performed Exploratory Data Analysis (EDA) to understand dataset distribution.
 - Applie data augmentation techniques (rotation, flipping, autocontrast, grayscale) to enhance model generalization.  
 - Split dataset into 3 subset: train, validation and test set. 
3. Model Training
 - Fine-tuned pre-trained models: SqueezeNet and GoogLeNet
 - Used Adam optimizer with a learning rate of 0.001.  
 - Employed custom early stopping to prevent overfitting.
4. Model Evaluation 
 - Evaluated performance using Classification Report (Precision, Recall, F1-score).  
 - Analyzed Confusion Matrix to assess misclassification patterns.  

## 📈 Some results & observations
- GoogLeNet achieved higher accuracy than SqueezeNet.
- Both models frequently misclassified the "trash" class, which can possibly be explained by the imbalanced distribution of data among classes. Additionally, the model often confused plastic with glass and cardboard with paper.
- Results table:

| Model       | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss | Time       |
|------------|---------------|--------------------|-----------|----------------|------------|
| SqueezeNet | 0.8739        | 0.8311             | 0.3557    | 0.5313         | 153.04 (s) |
| GoogLeNet  | 0.9774        | 0.9156             | 0.0856    | 0.2718         | 122.53 (s) |

## 💻 Technologies
The project is created with:
- Python 3.11.11
- libraries: numpy, pandas, matplotlib, seaborn, torch, sklearn.

Running the project: 
To run this project use Jupyter Notebook or Google Colab.
