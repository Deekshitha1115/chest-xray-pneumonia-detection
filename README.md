# Chest X-Ray Pneumonia Detection

A deep learning project that classifies chest X-ray images into two categories:

- NORMAL
- PNEUMONIA

The project uses a ResNet18-based deep learning model and provides a Streamlit web application for image classification.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- PIL
- ResNet18
- Deep Learning
- Computer Vision

## Project Workflow

1. Chest X-ray images are used for model training.
2. A ResNet18 model is trained for binary classification.
3. The trained model is saved as `pneumonia_model.pth`.
4. A Streamlit application loads the trained model.
5. Users can upload a chest X-ray image.
6. The application predicts whether the image is NORMAL or PNEUMONIA.

## Project Structure

```text
chest-xray-pneumonia-detection/
│
├── app.py
├── train.py
├── pneumonia_model.pth
├── output1.png
├── output2.png
├── output3.png
├── output4.png
├── Chest X-Ray.pdf
├── README.md
├── requirements.txt