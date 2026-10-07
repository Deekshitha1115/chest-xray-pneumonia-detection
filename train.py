import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader

print("Starting training...")

# Dataset paths
train_dir = "chest_xray/train"
test_dir = "chest_xray/test"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

print("Dataset loaded")

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

print("Dataloader ready")

# Load pretrained ResNet18 model (NEW METHOD)
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Change final layer for 2 classes
model.fc = nn.Linear(model.fc.in_features, 2)

print("Model loaded")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
epochs = 3

for epoch in range(epochs):
    running_loss = 0.0
    model.train()

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}  Loss: {running_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "pneumonia_model.pth")

print("Training finished")
print("Model saved as pneumonia_model.pth")