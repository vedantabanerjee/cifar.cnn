# CIFAR-10 Image Classification using Convolutional Neural Networks (CNNs)

**GitHub:** [vedantabanerjee](https://github.com/vedantabanerjee)  
**Twitter:** [0xr1sh1](https://twitter.com/0xr1sh1)

## Project Overview
This project aims to develop a model that can classify images from the CIFAR-10 dataset into one of ten categories. Using convolutional neural networks (CNNs), we will explore the dataset, preprocess the data, train the model, and evaluate its performance.

## Problem Statement
We want to create a predictive model that can accurately classify images into one of the ten classes in the CIFAR-10 dataset: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Each class contains 6,000 images, making it a balanced dataset for robust model training and evaluation.

## Data Attributes
To make accurate classifications, we will use the following information about each image:

- **Image data:** 32x32 pixels with RGB channels.
- **Image labels:** One of ten categories (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks).

## Process Outline

### 1. Data Collection and Preparation
The CIFAR-10 dataset is collected and loaded. Each image is 32x32 pixels in size and contains three color channels (RGB). The dataset is divided into 50,000 training images and 10,000 testing images.

### 2. Exploratory Data Analysis (EDA)
We start by exploring the data to understand it better. This involves visualizing some sample images and their corresponding labels to get an idea of the dataset's structure and distribution.

### 3. Data Preprocessing
Data preprocessing is crucial for improving the model's performance. This step involves:

- **Normalization:** Scaling the pixel values to a range of 0 to 1 to make the model training process more efficient.
- **Categorical Encoding:** Converting class labels into a one-hot encoded format, which is necessary for categorical classification tasks.

### 4. Model Architecture
We design a convolutional neural network (CNN) to classify the images. The architecture includes:

- **Convolutional Layers:** To extract features from the images.
- **Pooling Layers:** To reduce the dimensionality and computational load.
- **Dense Layers:** For classification based on the extracted features.
- **Dropout Layers:** To prevent overfitting by randomly setting a fraction of input units to 0 during training.

### 5. Model Training
The model is compiled and trained using the training dataset. Key aspects of training include:

- **Loss Function:** Categorical cross-entropy, suitable for multi-class classification tasks.
- **Optimizer:** RMSprop, which adjusts the learning rate dynamically.
- **Batch Size and Epochs:** The model is trained in batches over several epochs to gradually minimize the loss and improve accuracy.

### 6. Model Evaluation
After training, the model's performance is evaluated on the test dataset. Evaluation metrics include:

- **Accuracy:** The proportion of correctly classified images.
- **Confusion Matrix:** A detailed breakdown of correct and incorrect classifications across all classes.

### 7. Model Improvement: Image Augmentation
To enhance the model's performance, image augmentation techniques such as rotation, width shift, and flips are applied. This step increases the diversity of the training data and helps the model generalize better to unseen images.

### 8. Final Model Evaluation
The augmented model is retrained and evaluated again on the test dataset to measure the improved accuracy. This helps in understanding the impact of data augmentation on model performance.

## Conclusion
This project demonstrates how we can use CNNs to classify images in the CIFAR-10 dataset. By applying data augmentation techniques, the model's accuracy is improved, making it more robust and reliable for real-world applications.

## Evaluation Result
Our model achieved an accuracy of 72.85% before augmentation and improved with further training on augmented data. This demonstrates the effectiveness of image augmentation in enhancing model performance.

*Thank you, Vedanta Banerjee, 2024*
