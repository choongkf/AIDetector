import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    background-color: #f9f9f9;
}

.prediction-box {
    border: 3px solid;
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem 0;
    text-align: center;
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
    color: #007A6A;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #4ECDC4;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {}
    
    # Try to load RoBERTa model
    try:
        roberta_path = Path('../models/roberta_ai_detector')
        if roberta_path.exists():
            roberta_model = RobertaForSequenceClassification.from_pretrained(
                str(roberta_path), 
                num_labels=2
            )
            roberta_tokenizer = RobertaTokenizer.from_pretrained(str(roberta_path))
            roberta_model.to(device)
            roberta_model.eval()
            models['RoBERTa'] = {
                'model': roberta_model,
                'tokenizer': roberta_tokenizer,
                'loaded': True
            }
        else:
            models['RoBERTa'] = {'loaded': False}
    except Exception as e:
        models['RoBERTa'] = {'loaded': False, 'error': str(e)}
    
    # Try to load DistilBERT model
    try:
        distilbert_path = Path('../models/distilbert_ai_detector')
        if distilbert_path.exists():
            distilbert_model = DistilBertForSequenceClassification.from_pretrained(
                str(distilbert_path), 
                num_labels=2
            )
            distilbert_tokenizer = DistilBertTokenizer.from_pretrained(str(distilbert_path))
            distilbert_model.to(device)
            distilbert_model.eval()
            models['DistilBERT'] = {
                'model': distilbert_model,
                'tokenizer': distilbert_tokenizer,
                'loaded': True
            }
        else:
            models['DistilBERT'] = {'loaded': False}
    except Exception as e:
        models['DistilBERT'] = {'loaded': False, 'error': str(e)}
    
    return models, device

@st.cache_data
def load_results():
    """Load model comparison results"""
    results = {}
    
    # Load RoBERTa results
    try:
        with open('../models/roberta_results.json', 'r') as f:
            results['RoBERTa'] = json.load(f)
    except FileNotFoundError:
        results['RoBERTa'] = None
    
    # Load DistilBERT results
    try:
        with open('../models/distilbert_results.json', 'r') as f:
            results['DistilBERT'] = json.load(f)
    except FileNotFoundError:
        results['DistilBERT'] = None
    
    # Load comparison
    try:
        with open('../models/model_comparison.json', 'r') as f:
            results['comparison'] = json.load(f)
    except FileNotFoundError:
        results['comparison'] = None
    
    return results

def predict_text(text, model_name, models, device, max_length=512):
    """Predict if text is AI-generated or human-written"""
    if not models[model_name]['loaded']:
        return None, None, None
    
    model = models[model_name]['model']
    tokenizer = models[model_name]['tokenizer']
    
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return prediction.item(), probabilities[0].cpu().numpy(), inference_time

def main():
    # Load models and results
    models, device = load_models()
    results = load_results()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Text Detector</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
    Detect whether text is written by AI or humans using state-of-the-art transformer models
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Model Settings")
    
    # Model selection
    available_models = [name for name in models.keys() if models[name]['loaded']]
    
    if not available_models:
        st.error("❌ No trained models found! Please train the models first using the Jupyter notebooks.")
        st.stop()
    
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        available_models,
        help="Select which model to use for prediction"
    )
    
    # Model info
    st.sidebar.markdown("### 📊 Model Information")
    if results.get(selected_model):
        result = results[selected_model]
        st.sidebar.metric("Accuracy", f"{result['test_accuracy']:.1%}")
        st.sidebar.metric("F1-Score", f"{result['test_f1']:.3f}")
        if 'inference_time_ms' in result:
            st.sidebar.metric("Speed", f"{result['inference_time_ms']:.1f}ms")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Upload File", "Use Sample Text"],
            horizontal=True
        )
        
        text_to_analyze = ""
        
        if input_method == "Type/Paste Text":
            text_to_analyze = st.text_area(
                "Enter text to analyze:",
                height=200,
                placeholder="Paste your text here to check if it's AI-generated or human-written..."
            )
        
        elif input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt'],
                help="Upload a .txt file to analyze"
            )
            if uploaded_file is not None:
                text_to_analyze = str(uploaded_file.read(), "utf-8")
                st.text_area("Uploaded text:", text_to_analyze, height=150, disabled=True)
        
        elif input_method == "Use Sample Text":
            sample_type = st.selectbox(
                "Choose sample type:",
                ["AI-Generated Sample", "Human-Written Sample"]
            )
            
            if sample_type == "AI-Generated Sample":
                text_to_analyze = """Car-free cities have become a subject of increasing interest and debate in recent years, as urban areas around the world grapple with the challenges of congestion, pollution, and limited resources. The concept of a car-free city involves creating urban environments where private automobiles are either significantly restricted or completely banned, with a focus on alternative transportation methods and sustainable urban planning. This essay explores the benefits, challenges, and potential solutions associated with the idea of car-free cities."""
            else:
                text_to_analyze = """In conclusion Venus has its ups and downs but I do believe that it is worth studing despite the dangers that come with it . Also i do believe that we should send spacesrafts up there what are mre durable then the other ones that was sent there a long time ago because of the new tecnology that we have . I think that its possable that we could come up with more samples of different items ."""
            
            st.text_area("Sample text:", text_to_analyze, height=150, disabled=True)
    
    with col2:
        st.header("🎯 Prediction")
        
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if not text_to_analyze.strip():
                st.error("Please enter some text to analyze!")
            else:
                with st.spinner(f"Analyzing with {selected_model}..."):
                    prediction, probabilities, inference_time = predict_text(
                        text_to_analyze, selected_model, models, device
                    )
                
                if prediction is not None:
                    # Display prediction
                    if prediction == 1:
                        st.markdown(
                            '<div class="prediction-box ai-prediction">🤖 AI Generated</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="prediction-box human-prediction">👤 Human Written</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Confidence scores
                    st.subheader("📊 Confidence Scores")
                    
                    human_conf = probabilities[0] * 100
                    ai_conf = probabilities[1] * 100
                    
                    # Progress bars
                    st.metric("Human", f"{human_conf:.1f}%")
                    st.progress(human_conf / 100)
                    
                    st.metric("AI Generated", f"{ai_conf:.1f}%")
                    st.progress(ai_conf / 100)
                    
                    # Additional info
                    st.subheader("ℹ️ Analysis Info")
                    st.info(f"""
                    **Model Used:** {selected_model}
                    **Inference Time:** {inference_time:.1f}ms
                    **Text Length:** {len(text_to_analyze)} characters
                    **Word Count:** {len(text_to_analyze.split())} words
                    """)
                else:
                    st.error("Error during prediction. Please try again.")
    
    # Text statistics
    if text_to_analyze:
        st.header("📈 Text Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Characters", len(text_to_analyze))
        with col2:
            st.metric("Words", len(text_to_analyze.split()))
        with col3:
            st.metric("Sentences", len([s for s in text_to_analyze.split('.') if s.strip()]))
        with col4:
            avg_word_length = np.mean([len(word.strip('.,!?;:')) for word in text_to_analyze.split()])
            st.metric("Avg Word Length", f"{avg_word_length:.1f}")
    
    # Model Comparison
    st.header("🏆 Model Comparison")
    
    if results.get('comparison'):
        comparison = results['comparison']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Metrics")
            if 'performance_comparison' in comparison:
                perf_data = comparison['performance_comparison']
                df = pd.DataFrame(perf_data)
                df = df.set_index('Model')
                st.dataframe(df.round(4))
        
        with col2:
            st.subheader("💡 Recommendations")
            if 'recommendations' in comparison:
                rec = comparison['recommendations']
                st.info(f"**Best Accuracy:** {rec.get('best_accuracy', 'N/A')}")
                st.info(f"**Best Speed:** {rec.get('best_speed', 'N/A')}")
                st.success(f"**Recommended for Production:** {rec.get('recommended_for_production', 'N/A')}")
    
    else:
        st.info("Model comparison data not available. Run the comparison notebook to see detailed metrics.")
    
    # Model Status
    st.header("🔧 Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RoBERTa Model")
        if models['RoBERTa']['loaded']:
            st.success("✅ Loaded successfully")
            if results.get('RoBERTa'):
                st.write(f"**Accuracy:** {results['RoBERTa']['test_accuracy']:.1%}")
                st.write(f"**Parameters:** {results['RoBERTa'].get('model_parameters', 'N/A'):,}")
        else:
            st.error("❌ Not loaded")
            st.write("Train the RoBERTa model using notebook 02")
    
    with col2:
        st.subheader("DistilBERT Model")
        if models['DistilBERT']['loaded']:
            st.success("✅ Loaded successfully")
            if results.get('DistilBERT'):
                st.write(f"**Accuracy:** {results['DistilBERT']['test_accuracy']:.1%}")
                st.write(f"**Parameters:** {results['DistilBERT'].get('model_parameters', 'N/A'):,}")
        else:
            st.error("❌ Not loaded")
            st.write("Train the DistilBERT model using notebook 03")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
    🤖 AI Text Detector | Built with Streamlit, PyTorch & Transformers<br>
    Models: RoBERTa-base & DistilBERT-base | Dataset: 23,276 samples (balanced)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
