"""
AI Text Detection Web App

This Streamlit application allows users to detect AI-generated text using trained transformer models.
It provides both quick detection and detailed analysis reports showing how predictions are made.
"""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import models
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)

# Import prediction report functionality
from prediction_report import show_prediction_report_page

# Page configuration
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for home page
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.model-card {
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    background: linear-gradient(135deg, #32CD32 0%, #006400 100%);
}

.prediction-box {
    border: 3px solid;
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
    font-size: 1.5rem;
    font-weight: bold;
}

.ai-prediction {
    border-color: #FF6B6B;
    background-color: #FFE5E5;
    color: #CC0000;
}

.human-prediction {
    border-color: #4ECDC4;
    background-color: #E5F9F7;
    color: #008B8B;
}

.feature-metric {
    background-color: #E5F9F7;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model loading functions
def get_model_path(model_name):
    """Get the correct model path, checking multiple possible locations"""
    possible_paths = [
        Path(f"models/{model_name}"),  # If running from project root
        Path(f"../models/{model_name}"),  # If running from streamlit_app folder
        Path(f"../AIDetector/models/{model_name}"),  # If running from parent directory
        Path(__file__).parent.parent / "models" / model_name,  # Relative to this file
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def debug_model_paths():
    """Debug function to show model search paths"""
    st.write("🔍 **Model Path Debug Information:**")
    
    current_dir = Path.cwd()
    script_dir = Path(__file__).parent
    
    st.write(f"- **Current working directory:** `{current_dir}`")
    st.write(f"- **Script directory:** `{script_dir}`")
    
    model_names = ["roberta_ai_detector", "distilbert_ai_detector"]
    
    for model_name in model_names:
        st.write(f"\n**Looking for {model_name}:**")
        
        possible_paths = [
            Path(f"models/{model_name}"),
            Path(f"../models/{model_name}"),
            Path(f"../AIDetector/models/{model_name}"),
            Path(__file__).parent.parent / "models" / model_name,
        ]
        
        for i, path in enumerate(possible_paths, 1):
            exists = "✅" if path.exists() else "❌"
            st.write(f"  {i}. {exists} `{path.resolve()}`")

@st.cache_resource
def load_roberta_model():
    """Load RoBERTa model"""
    try:
        model_path = get_model_path("roberta_ai_detector")
        
        if model_path and model_path.exists():
            tokenizer = RobertaTokenizer.from_pretrained(str(model_path))
            model = RobertaForSequenceClassification.from_pretrained(str(model_path))
            model.to(device)
            model.eval()
            st.success(f"✅ RoBERTa model loaded from: {model_path}")
            return tokenizer, model, True
        else:
            checked_paths = [
                "models/roberta_ai_detector",
                "../models/roberta_ai_detector", 
                "../AIDetector/models/roberta_ai_detector"
            ]
            st.warning(f"❌ RoBERTa model not found. Checked paths: {checked_paths}")
            return None, None, False
    except Exception as e:
        st.error(f"❌ Error loading RoBERTa model: {str(e)}")
        return None, None, False

@st.cache_resource
def load_distilbert_model():
    """Load DistilBERT model"""
    try:
        model_path = get_model_path("distilbert_ai_detector")
        
        if model_path and model_path.exists():
            tokenizer = DistilBertTokenizer.from_pretrained(str(model_path))
            model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
            model.to(device)
            model.eval()
            st.success(f"✅ DistilBERT model loaded from: {model_path}")
            return tokenizer, model, True
        else:
            checked_paths = [
                "models/distilbert_ai_detector",
                "../models/distilbert_ai_detector", 
                "../AIDetector/models/distilbert_ai_detector"
            ]
            st.warning(f"❌ DistilBERT model not found. Checked paths: {checked_paths}")
            return None, None, False
    except Exception as e:
        st.error(f"❌ Error loading DistilBERT model: {str(e)}")
        return None, None, False

def predict_text(text, model, tokenizer):
    """Make prediction on text"""
    if len(text.strip()) < 10:
        return None, "Text too short for reliable prediction"
    
    try:
        # Tokenize
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = torch.max(predictions, dim=1)[0].item()
            predicted_class = torch.argmax(predictions, dim=1).item()
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'human_prob': float(predictions[0][0]),
            'ai_prob': float(predictions[0][1])
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def show_home_page(models):
    """Display the main detection interface"""
    
    st.markdown('<h1 class="main-header">🤖 AI Text Detector</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
    Detect if text was written by AI or humans using state-of-the-art transformer models
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 Detailed Analysis Report", type="secondary", use_container_width=True):
            st.session_state.page = "report"
            st.rerun()
    
    st.divider()
    
    # Model status
    st.header("🧠 Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card">
        <h3>🔴 RoBERTa Base</h3>
        <p><strong>Parameters:</strong> 125M</p>
        <p><strong>Training:</strong> Optimized for accuracy</p>
        <p><strong>Status:</strong> {}</p>
        </div>
        """.format("✅ Loaded" if models['RoBERTa']['loaded'] else "❌ Not Available"), 
        unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
        <h3>🔵 DistilBERT Base</h3>
        <p><strong>Parameters:</strong> 66M</p>
        <p><strong>Training:</strong> Optimized for speed</p>
        <p><strong>Status:</strong> {}</p>
        </div>
        """.format("✅ Loaded" if models['DistilBERT']['loaded'] else "❌ Not Available"), 
        unsafe_allow_html=True)
    
    st.divider()
    
def main():
    """Main application function"""
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Load models
    roberta_tokenizer, roberta_model, roberta_loaded = load_roberta_model()
    distilbert_tokenizer, distilbert_model, distilbert_loaded = load_distilbert_model()
    
    # Store models in a dictionary
    models = {
        'RoBERTa': {
            'tokenizer': roberta_tokenizer,
            'model': roberta_model,
            'loaded': roberta_loaded
        },
        'DistilBERT': {
            'tokenizer': distilbert_tokenizer,
            'model': distilbert_model,
            'loaded': distilbert_loaded
        }
    }
    
    # Page navigation
    if st.session_state.page == "home":
        show_home_page(models)
    elif st.session_state.page == "report":
        show_prediction_report_page(models, device)

if __name__ == "__main__":
    main()
