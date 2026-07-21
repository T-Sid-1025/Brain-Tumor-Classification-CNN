import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from tensorflow.keras import layers, models

st.set_page_config(
    page_title="NeuroVision Clinical - Brain Tumor Analysis",
    page_icon="🧬",
    layout="wide"
)

IMG_SIZE = (150, 150)

@st.cache_resource
def load_model():
    try:
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3), padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(256, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(r'D:\brain_tumor_mri\best_brain_tumor_model.h5')
        return model
    except:
        return None

def preprocess(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img).astype("float32") / 255.0, axis=0)

# SIDEBAR
with st.sidebar:
    st.title("NeuroVision Clinical")
    st.write("Brain Tumor Analysis System")
    st.divider()
    
    st.subheader("System Specifications")
    st.metric("Model Accuracy", "76.74%")
    st.metric("Training Cases", "3,064")
    st.metric("Optimal Epoch", "29")
    st.metric("System Status", "ACTIVE")
    
    st.divider()
    
    st.subheader("Clinical Classifications")
    st.write("• TUMOR - Brain tumor detected")
    st.write("• NORMAL - No tumor identified")
    st.write("• INCONCLUSIVE - Requires radiologist review")
    
    st.divider()
    
    st.subheader("Project Information")
    st.write("Institute: Innomatics Research & Labs")
    st.write("Type: CNN Capstone - Medical AI")
    st.write("Version: 1.0 Clinical")

# MAIN HEADER
st.title("NeuroVision Clinical")
st.write("Automated Brain Tumor MRI Analysis System")

# METRICS
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "76.74%")
col2.metric("Cases", "3,064")
col3.metric("Epoch", "29")
col4.metric("Status", "READY")

st.divider()

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Analysis", "Performance", "Guidelines"])

# TAB 1: DASHBOARD
with tab1:
    st.header("Clinical Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Architecture")
        st.write("• Type: Convolutional Neural Network")
        st.write("• Conv Blocks: 4")
        st.write("• Input: 150×150 RGB")
        st.write("• Output Classes: 3")
    
    with col2:
        st.subheader("Dataset Distribution")
        st.write("• Tumor Cases: 708 (23.1%)")
        st.write("• Normal Cases: 1,426 (46.5%)")
        st.write("• Inconclusive: 930 (30.4%)")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", "77%")
    col2.metric("Recall", "77%")
    col3.metric("F1-Score", "0.76")

# TAB 2: ANALYSIS
with tab2:
    st.header("Patient MRI Analysis")
    
    col_info, col_result = st.columns([0.4, 0.6])
    
    with col_info:
        st.subheader("Patient Information")
        patient_id = st.text_input("Patient ID", placeholder="MR-2024-XXXXX")
        scan_date = st.date_input("Scan Date")
        radiologist = st.text_input("Radiologist Name", placeholder="Dr. Name")
        
        st.divider()
        st.subheader("Upload MRI")
        uploaded_file = st.file_uploader("Select MRI Scan", type=["jpg", "jpeg", "png"])
    
    with col_result:
        if uploaded_file:
            st.subheader("Uploaded MRI Scan")
            img = Image.open(uploaded_file)
            
            # Fixed image size
            st.image(img, caption="MRI Scan (150x150px)", width=350)
            
            st.divider()
            
            st.subheader("AI Analysis Result")
            
            with st.spinner("Processing MRI..."):
                model = load_model()
                if model:
                    pred = model.predict(preprocess(img), verbose=0)[0]
                    idx = np.argmax(pred)
                    conf = pred[idx]
                    
                    class_names = {0: "NORMAL", 1: "TUMOR DETECTED", 2: "INCONCLUSIVE"}
                    pred_name = class_names[idx]
                    
                    if idx == 1:
                        st.error(f"⚠️ FINDING: TUMOR DETECTED\nConfidence: {conf*100:.1f}%")
                    elif idx == 0:
                        st.success(f"✓ FINDING: NORMAL\nConfidence: {conf*100:.1f}%")
                    else:
                        st.warning(f"⚠️ FINDING: INCONCLUSIVE\nConfidence: {conf*100:.1f}%")
    
    st.divider()
    
    if uploaded_file and model:
        st.subheader("Probability Distribution")
        
        # Fixed graph size
        fig, ax = plt.subplots(figsize=(12, 5))
        classes = ["NORMAL", "TUMOR", "INCONCLUSIVE"]
        colors = ['#388e3c', '#d32f2f', '#f57c00']
        ax.barh(classes, pred*100, color=colors, edgecolor='black', linewidth=2, height=0.6)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence Score (%)", fontsize=14, fontweight='bold')
        ax.set_title("AI Classification Confidence Distribution", fontsize=16, fontweight='bold')
        
        for i, v in enumerate(pred*100):
            ax.text(v + 1.5, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=12)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# TAB 3: PERFORMANCE
with tab3:
    st.header("System Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        if os.path.exists(r"D:\brain_tumor_mri\confusion_matrix.png"):
            st.image(r"D:\brain_tumor_mri\confusion_matrix.png", width=400)
    
    with col2:
        st.subheader("Training History")
        if os.path.exists(r"D:\brain_tumor_mri\training_history.png"):
            st.image(r"D:\brain_tumor_mri\training_history.png", width=400)
    
    st.divider()
    
    st.subheader("Clinical Performance Metrics")
    metrics_data = {
        "Class": ["Normal", "Tumor", "Other"],
        "Precision": ["76%", "71%", "88%"],
        "Recall": ["50%", "86%", "86%"],
        "F1-Score": [0.60, 0.78, 0.87],
        "Support": [117, 209, 134]
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

# TAB 4: GUIDELINES
with tab4:
    st.header("Clinical Usage Guidelines")
    
    st.subheader("System Purpose")
    st.write("NeuroVision Clinical is a clinical decision support system designed to assist radiologists in brain tumor detection from MRI scans.")
    
    st.subheader("Usage Workflow")
    st.write("1. Load high-quality MRI brain scan")
    st.write("2. Review AI-generated analysis")
    st.write("3. Cross-reference with clinical history")
    st.write("4. Perform independent radiologist assessment")
    st.write("5. Document final clinical diagnosis")
    
    st.subheader("Technical Specifications")
    st.write("• Architecture: Convolutional Neural Network")
    st.write("• Input: RGB MRI Images (150×150)")
    st.write("• Processing Time: <1 second per scan")
    
    st.subheader("Limitations")
    st.write("• Accuracy: 76.74% (not 100%)")
    st.write("• Requires radiologist verification")
    st.write("• Best used with high-resolution MRI scans")
    
    st.error("⚠️ CRITICAL DISCLAIMER")
    st.write("This system is a clinical decision support tool only. Results must be verified by qualified radiologist. This is NOT a clinical diagnosis.")

st.divider()
st.write("© 2026 NeuroVision Clinical | Innomatics Research & Labs | For Research and Clinical Support Use Only")