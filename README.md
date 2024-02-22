# Potato-Leaf-Disease-Classification-Model

### Introduction
This project focuses on the development of a deep learning model to classify diseases affecting potato and tomato leaves. The ability to detect plant diseases early is crucial for farmers to prevent crop damage and ensure high yields. By leveraging deep learning techniques, we aim to create a model that can accurately identify various diseases in potato and tomato plants.

### Dataset
The dataset used for training and testing the model consists of images of potato and tomato leaves affected by different diseases. The dataset is sourced from [Kaggle](https://www.kaggle.com/arjuntejaswi/plant-village) and includes classes such as "Potato___Early_blight," "Potato___Late_blight," "Potato___healthy," and will be extended to include classes related to tomato leaf diseases.

### Model Architecture
The model architecture is based on a Convolutional Neural Network (CNN), which is well-suited for image classification tasks. The CNN is comprised of multiple convolutional and pooling layers followed by fully connected layers and a softmax output layer. The model architecture is optimized for high accuracy in classifying potato and tomato leaf diseases.

### Workflow
1. **Data Preparation**: The dataset is preprocessed and divided into training, validation, and testing sets. Data augmentation techniques are applied to increase the variability of training samples and prevent overfitting.

2. **Model Training**: The CNN model is trained on the training dataset using the Adam optimizer and Sparse Categorical Crossentropy loss function. During training, the model learns to classify images into various disease categories.

3. **Model Evaluation**: The trained model is evaluated on the validation and testing datasets to assess its performance in terms of accuracy and loss. Visualizations such as accuracy and loss curves are generated to analyze the model's training progress.

4. **Inference**: The trained model is used to make predictions on sample images of potato and tomato leaves. These predictions help demonstrate the model's ability to correctly identify diseases in unseen images.

### Future Work
In the future, we plan to:
- Expand the dataset to include more classes of tomato leaf diseases.
- Further optimize the model architecture and hyperparameters to improve performance.
- Explore techniques for model interpretability to understand the factors influencing disease classification decisions.
- Deploy the trained model as a user-friendly tool for farmers to diagnose plant diseases in real-time.

By developing an accurate and reliable model for potato and tomato leaf disease classification, we aim to empower farmers with valuable insights to protect their crops and ensure food security.

---
