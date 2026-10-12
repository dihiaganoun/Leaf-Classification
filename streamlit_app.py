# streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# === Settings ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
NUM_CLASSES =  2
CLASS_NAMES = ['dente', 'lisse']

# === Transform for inference ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load model ===
@st.cache_resource
def load_model():
    model = models.resnext101_32x8d(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("resnext101_leaves.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# === Streamlit UI ===
st.title("🍃 Leaf Classifier")
st.write("Upload an image of a leaf and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Use two columns: left for image, right for prediction
    col1, col2 = st.columns([1, 1])  # adjust ratio if needed
    
    with col1:
        st.image(image, caption='Uploaded Leaf', use_container_width=False, width=250)  # smaller image
    
    with col2:
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        st.markdown("### Prediction")
        st.write(f"**Class:** {CLASS_NAMES[pred_idx]}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
