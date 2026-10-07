import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

st.title("🩺 Chest X-ray Pneumonia Detection")

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("pneumonia_model.pth", map_location=torch.device('cpu')))
model.eval()

classes = ["NORMAL", "PNEUMONIA"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # FIXED LINE
    st.image(image, caption="Uploaded Image", width=300)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    st.success(f"Prediction: {classes[predicted.item()]}")
    st.info(f"Confidence: {confidence.item()*100:.2f}%")