import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import os

# ==========================================
# ⚙️ CONFIGURATION - UPDATE THESE PATHS!
# ==========================================
# 1. Path to your trained model file
MODEL_PATH = "saved_models/best_model.pth" 

# 2. Path to the folder containing your celebrity subfolders 
# (e.g., './processed_faces' or './dataset/bollywood_celeb_faces2')
# The app will scan this to get the actor names alphabetically!
DATA_DIR = "./processed_faces" 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(page_title="Bollywood Celeb Lookalike", page_icon="🎬", layout="centered")

# --- LOAD RESOURCES (Cached) ---
@st.cache_resource
def load_class_mapping(data_dir):
    """Scans the dataset directory to reconstruct the class names alphabetically."""
    if not os.path.exists(data_dir):
        st.error(f"Dataset directory '{data_dir}' not found. Please update DATA_DIR in the code.")
        st.stop()
        
    # Get all folder names inside the directory and sort them alphabetically
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Create the mapping {0: 'Aamir Khan', 1: 'Abhay Deol', ...}
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}
    return idx_to_class, len(class_names)

@st.cache_resource
def load_models(num_classes):
    """Loads MTCNN for face detection and EfficientNet for classification."""
    # 1. Face Extractor (MTCNN)
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device=DEVICE)
    
    # 2. Re-create Model Architecture
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # 3. Load Weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Handle if your checkpoint saved the model directly vs inside a 'state_dict' key
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(DEVICE)
    model.eval() 
    return mtcnn, model

# --- PREPROCESSING ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- INFERENCE LOGIC ---
def predict_top_5(image, mtcnn, model, idx_to_class):
    # Extract face
    face = mtcnn(image)
    if face is None:
        return None, None 
        
    # Convert PyTorch tensor back to PIL Image for torchvision transforms
    face_np = face.permute(1, 2, 0).cpu().numpy()
    face_np = np.clip(face_np, 0, 255).astype(np.uint8)
    face_pil = Image.fromarray(face_np)

    # Preprocess and prepare for model
    input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Get Top 5
    top_prob, top_indices = torch.topk(probabilities, 5)
    top_prob = top_prob.cpu().numpy() * 100 
    top_indices = top_indices.cpu().numpy()
    
    results = []
    for i in range(5):
        celeb_name = idx_to_class[top_indices[i]]
        results.append((celeb_name, top_prob[i]))
        
    return face_pil, results

# ==========================================
# STREAMLIT UI
# ==========================================
st.title("🎬 Which Bollywood Celeb Do You Look Like?")
st.write("Upload a photo or take a picture to find your top 5 Bollywood celebrity matches!")

# Initialize
idx_to_class, num_classes = load_class_mapping(DATA_DIR)
mtcnn, model = load_models(num_classes)

# UI Tabs
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Take a Photo"])

image_to_process = None

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert('RGB')
        st.image(image_to_process, caption="Uploaded Image", use_column_width=True)

with tab2:
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image_to_process = Image.open(camera_file).convert('RGB')

# Processing
if image_to_process is not None:
    st.markdown("---")
    with st.spinner("Detecting face and analyzing features... 🤖"):
        cropped_face, predictions = predict_top_5(image_to_process, mtcnn, model, idx_to_class)
        
    if cropped_face is None:
        st.error("🚨 No face detected! Please try another image with a clear view of a face.")
    else:
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Extracted Face:**")
            st.image(cropped_face, use_column_width=True)
            
        with col2:
            st.write("**Top 5 Matches:**")
            for rank, (name, probability) in enumerate(predictions, 1):
                st.markdown(f"**{rank}. {name}** - *{probability:.2f}%*")
                st.progress(int(probability))