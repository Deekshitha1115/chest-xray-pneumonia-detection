# 🩺 Chest X-Ray Pneumonia Detection

## 📌 Project Overview

This project is a deep learning-based application that detects pneumonia from chest X-ray images.

The project uses a pre-trained ResNet18 convolutional neural network (CNN) model and transfer learning to classify chest X-ray images into two categories:

- NORMAL
- PNEUMONIA

The trained model is integrated with a Streamlit web application where users can upload a chest X-ray image and receive a prediction along with the model confidence score.

---

## 🎯 Project Objectives

- Detect pneumonia from chest X-ray images.
- Build and train a deep learning image classification model.
- Use ResNet18 for image classification.
- Save the trained PyTorch model.
- Build a user-friendly Streamlit web application.
- Allow users to upload X-ray images and receive predictions.
- Display the prediction and confidence score.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Deep Learning

- PyTorch
- TorchVision
- ResNet18
- Convolutional Neural Networks (CNN)

### Image Processing

- Pillow (PIL)

### Web Application

- Streamlit

### Development Tools

- Visual Studio Code
- Git
- GitHub

---

## 🧠 Model Architecture

The project uses **ResNet18**, a convolutional neural network architecture available through TorchVision.

The final fully connected layer of ResNet18 is modified to classify the images into two classes:

```text
NORMAL
PNEUMONIA