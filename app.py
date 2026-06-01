import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import os
import time

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Pneumonia Diagnosis Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Header Styling */
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        font-size: 1.2rem;
        color: #7A7A7A;
        margin-bottom: 2.5rem;
    }
    
    /* Dashboard Summary Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 75, 75, 0.4);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0A0A0;
        margin-top: 0.5rem;
    }

    /* Model Detections Grid Card Styling */
    .model-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    .model-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .status-detected {
        background-color: rgba(244, 67, 54, 0.15);
        color: #FF5252;
        border: 1px solid rgba(244, 67, 54, 0.3);
        border-radius: 30px;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-normal {
        background-color: rgba(76, 175, 80, 0.15);
        color: #69F0AE;
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 30px;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    .status-missing {
        background-color: rgba(120, 120, 120, 0.1);
        color: #9E9E9E;
        border: 1px solid rgba(120, 120, 120, 0.2);
        border-radius: 30px;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Disclaimer card styling */
    .disclaimer-card {
        background: rgba(255, 152, 0, 0.05);
        border-left: 4px solid #FF9800;
        border-radius: 4px 12px 12px 4px;
        padding: 1.5rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & UTILITIES
# ==========================================
MODEL_NAMES = [
    "DenseNet121",
    "EfficientNetB0",
    "MobileNetV2",
    "ResNet50",
    "VGG16",
    "InceptionV3"
]

POSSIBLE_DIRS = [
    "pneumonia_output/models",
    "models",
    "./",
    "/content/pneumonia_output/models",
]

def find_model_path(model_name):
    """Searches for model file in multiple potential folders."""
    for d in POSSIBLE_DIRS:
        path = os.path.join(d, f"{model_name}.keras")
        if os.path.exists(path):
            return path
    return None

@st.cache_resource
def load_single_model(model_name):
    """Loads a single Keras model with correct custom objects (lambda layers)."""
    path = find_model_path(model_name)
    if not path:
        return None
        
    custom_objects = {}
    if "InceptionV3" in model_name:
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        custom_objects = {"preprocess_input": preprocess_input}
    elif "ResNet50" in model_name:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        custom_objects = {"preprocess_input": preprocess_input}
    elif "VGG16" in model_name:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        custom_objects = {"preprocess_input": preprocess_input}
    elif "DenseNet121" in model_name:
        from tensorflow.keras.applications.densenet import preprocess_input
        custom_objects = {"preprocess_input": preprocess_input}
    elif "MobileNetV2" in model_name:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        custom_objects = {"preprocess_input": preprocess_input}
        
    try:
        model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}")
        return None

def preprocess_image(uploaded_file, target_size=(224, 224)):
    """Preprocesses a PIL image for model consumption."""
    img = Image.open(uploaded_file).convert('RGB')
    # Use Lanczos resampling matching the pipeline training
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/lung.png", width=120)
    st.markdown("### **System Status**")
    
    # Load and check models status
    loaded_models = {}
    for name in MODEL_NAMES:
        path = find_model_path(name)
        if path:
            st.success(f"✔️ {name}: Available")
            # Load with cache
            model = load_single_model(name)
            if model:
                loaded_models[name] = model
        else:
            st.warning(f"⚠️ {name}: Not found")
            
    st.markdown("---")
    st.markdown("### **Deployment Instructions**")
    st.info("Place your trained `.keras` files inside a folder named `models` or `pneumonia_output/models` in the same directory as this script.")

# ==========================================
# MAIN PAGE INTERFACE
# ==========================================
st.markdown('<div class="header-title">Pneumonia Diagnosis Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Ensemble Chest Radiograph Diagnostics with 6 Deep Transfer Learning Models</div>', unsafe_allow_html=True)

# Image upload section
uploaded_file = st.file_uploader("Upload Chest Radiograph (X-Ray in JPEG/PNG format):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    pil_img, img_array = preprocess_image(uploaded_file)
    
    # Grid Layout: Left Column = Image, Right Column = Ensemble Dashboard
    col_img, col_dashboard = st.columns([1, 1])
    
    with col_img:
        st.markdown("### **Input Radiograph**")
        st.image(pil_img, use_column_width=True, caption="Uploaded Chest X-Ray")
        
    with col_dashboard:
        st.markdown("### **Ensemble Dashboard**")
        
        # We run the predictions dynamically
        predictions = {}
        with st.spinner("Processing image across all active models..."):
            for name, model in loaded_models.items():
                pred = model.predict(img_array, verbose=0)
                prob_normal = pred[0][0]
                prob_pneumonia = pred[0][1]
                label = "PNEUMONIA" if prob_pneumonia > prob_normal else "NORMAL"
                confidence = prob_pneumonia if label == "PNEUMONIA" else prob_normal
                predictions[name] = {
                    "label": label,
                    "prob_pneumonia": prob_pneumonia,
                    "prob_normal": prob_normal,
                    "confidence": confidence
                }
        
        # Calculate ensemble metrics
        total_models = len(MODEL_NAMES)
        active_models = len(predictions)
        detected_count = sum(1 for p in predictions.values() if p["label"] == "PNEUMONIA")
        
        # Metric Cards Layout
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{detected_count} / {active_models}</div>
                <div class="metric-label">Consensus Detections</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            ensemble_risk = (detected_count / active_models * 100) if active_models > 0 else 0
            risk_color = "#FF4B4B" if ensemble_risk > 50 else "#69F0AE"
            st.markdown(f"""
            <div class="metric-card" style="border-color: {risk_color}33;">
                <div class="metric-value" style="color: {risk_color};">{ensemble_risk:.1f}%</div>
                <div class="metric-label">Computed Clinical Risk</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Brief diagnostic commentary based on consensus
        if active_models == 0:
            st.error("No model files found. Please see the sidebar instructions to upload your trained models.")
        elif detected_count == 0:
            st.success("Consensus verdict: **NORMAL**. All active models evaluated this lung scan as healthy (Normal).")
        elif detected_count == active_models:
            st.error("Consensus verdict: **HIGH RISK OF PNEUMONIA**. All active models detected lung consolidations associated with pneumonia.")
        else:
            st.warning(f"Consensus verdict: **DISCREPANT DETECTION**. {detected_count} out of {active_models} models detected pneumonia patterns. Clinical evaluation recommended.")

    # ==========================================
    # DETAILED DETECTIONS GRID
    # ==========================================
    st.markdown("---")
    st.markdown("### **Individual Model Analysis**")
    
    # 3x2 Grid for the 6 models
    cols = st.columns(3)
    
    for idx, name in enumerate(MODEL_NAMES):
        col = cols[idx % 3]
        with col:
            if name in predictions:
                pred_data = predictions[name]
                label = pred_data["label"]
                prob_pneumonia = pred_data["prob_pneumonia"]
                confidence = pred_data["confidence"]
                
                status_class = "status-detected" if label == "PNEUMONIA" else "status-normal"
                status_text = "PNEUMONIA DETECTED" if label == "PNEUMONIA" else "NORMAL (HEALTHY)"
                
                col.markdown(f"""
                <div class="model-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <span style="font-weight: 600; font-size: 1.15rem;">{name}</span>
                        <span class="{status_class}">{status_text}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #A0A0A0; margin-bottom: 0.5rem;">
                        Confidence Score: <b>{confidence*100:.2f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Visual confidence bar
                col.progress(float(confidence))
                
            else:
                col.markdown(f"""
                <div class="model-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <span style="font-weight: 600; font-size: 1.15rem; color: #7A7A7A;">{name}</span>
                        <span class="status-missing">FILE NOT FOUND</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #7A7A7A;">
                        To evaluate with this model, place <code>{name}.keras</code> in the models directory.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
else:
    # User guidance on landing
    st.info("Please upload a chest radiograph to start the automatic diagnosis diagnostics.")

# ==========================================
# MEDICAL DISCLAIMER
# ==========================================
st.markdown("""
<div class="disclaimer-card">
    <div style="font-weight: 600; font-size: 1.1rem; color: #FF9800; margin-bottom: 0.5rem; display: flex; align-items: center;">
        <span style="margin-right: 0.5rem;">⚠️</span> MEDICAL DISCLAIMER & SAFETY INFORMATION
    </div>
    <div style="font-size: 0.95rem; color: #DFDFDF; line-height: 1.6;">
        <ul>
            <li><b>Not a Diagnostic Tool:</b> This software is a research prototype powered by transfer learning neural networks and is <b>NOT</b> a substitute for professional medical advice, clinical diagnosis, or treatment.</li>
            <li><b>Clinical Workflow:</b> All model outputs, diagnostic consensus, and risk predictions are solely for assistant screening. Chest X-rays must always be reviewed by a board-certified radiologist or trained medical practitioner.</li>
            <li><b>False Negatives & Specificity Trade-off:</b> To maximize clinical safety, these models have been optimized for high sensitivity (Recall > 99%). This means the models are designed to minimize missed cases, which can occasionally lead to false positive alerts on healthy scans.</li>
            <li><b>Technical Limit:</b> Predictions are generated using a standard model resolution of 224x224 pixels. Subtle consolidations or early-stage infiltrates may require full high-resolution clinical DICOM viewers.</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)
