"""
AI Detection Dual Model Analysis Report Module

This module provides comprehensive analysis comparing both RoBERTa and DistilBERT models
for AI text detection, showing detailed predictions, model agreement, and decision patterns.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path


class DualModelAIDetectionReport:
    """
    Generates comprehensive analysis reports comparing both AI                        st.markdown('<h3 class="roberta-header">🔴 RoBERTa Analysis</h3>', unsafe_allow_html=True)ction models
    """
    
    def __init__(self, models, device):
        self.models = models
        self.device = device
        
    def analyze_text_features(self, text):
        """Extract linguistic features from text"""
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'avg_word_length': np.mean([len(word.strip('.,!?;:"()[]')) for word in words]) if words else 0,
            'unique_words': len(set(word.lower().strip('.,!?;:"()[]') for word in words)),
            'lexical_diversity': len(set(word.lower() for word in words)) / max(len(words), 1),
            'punctuation_ratio': sum(1 for char in text if char in '.,!?;:') / max(len(text), 1),
            'uppercase_ratio': sum(1 for char in text if char.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for char in text if char.isdigit()) / max(len(text), 1)
        }
        
        # Complex word analysis (words with 3+ syllables)
        complex_words = [word for word in words if self._count_syllables(word) >= 3]
        features['complex_word_ratio'] = len(complex_words) / max(len(words), 1)
        
        # Sentence length variation
        sentence_lengths = [len(sent.split()) for sent in sentences]
        if sentence_lengths:
            features['sentence_length_std'] = np.std(sentence_lengths)
            features['max_sentence_length'] = max(sentence_lengths)
            features['min_sentence_length'] = min(sentence_lengths)
        else:
            features['sentence_length_std'] = 0
            features['max_sentence_length'] = 0
            features['min_sentence_length'] = 0
            
        return features
    
    def _count_syllables(self, word):
        """Simple syllable counting heuristic"""
        word = word.lower().strip('.,!?;:"()[]')
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
                
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
            
        return max(syllable_count, 1)
    
    def get_dual_model_predictions(self, text):
        """Get predictions from both models for comparison"""
        results = {}
        
        for model_name in ['RoBERTa', 'DistilBERT']:
            if not self.models[model_name]['loaded']:
                results[model_name] = None
                continue
                
            model = self.models[model_name]['model']
            tokenizer = self.models[model_name]['tokenizer']
            
            # Tokenize
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
            # Calculate confidence metrics
            confidence_scores = probabilities[0].cpu().numpy()
            prediction = np.argmax(confidence_scores)
            confidence = float(np.max(confidence_scores))
            uncertainty = 1 - confidence
            
            # Calculate entropy (measure of uncertainty)
            entropy = -np.sum(confidence_scores * np.log2(confidence_scores + 1e-10))
            
            results[model_name] = {
                'prediction': prediction,
                'human_probability': float(confidence_scores[0]),
                'ai_probability': float(confidence_scores[1]),
                'confidence': confidence,
                'uncertainty': uncertainty,
                'entropy': entropy,
                'raw_logits': logits[0].cpu().numpy().tolist(),
                'tokens': tokenizer.tokenize(text[:200])[:20]  # First 20 tokens for display
            }
        
        return results
    
    def analyze_model_agreement(self, predictions):
        """Analyze agreement between the two models"""
        if not all(predictions.values()):
            return None
            
        roberta_pred = predictions['RoBERTa']['prediction']
        distilbert_pred = predictions['DistilBERT']['prediction']
        
        # Check if models agree
        agreement = roberta_pred == distilbert_pred
        
        # Calculate confidence difference
        roberta_conf = predictions['RoBERTa']['confidence']
        distilbert_conf = predictions['DistilBERT']['confidence']
        confidence_diff = abs(roberta_conf - distilbert_conf)
        
        # Calculate prediction probability differences
        human_prob_diff = abs(
            predictions['RoBERTa']['human_probability'] - 
            predictions['DistilBERT']['human_probability']
        )
        ai_prob_diff = abs(
            predictions['RoBERTa']['ai_probability'] - 
            predictions['DistilBERT']['ai_probability']
        )
        
        return {
            'agreement': agreement,
            'confidence_difference': confidence_diff,
            'human_probability_difference': human_prob_diff,
            'ai_probability_difference': ai_prob_diff,
            'consensus_strength': 'Strong' if confidence_diff < 0.1 and agreement else 
                                'Moderate' if confidence_diff < 0.2 and agreement else 'Weak'
        }

    def generate_ai_indicators(self, text, features):
        """Generate AI detection indicators"""
        indicators = []
        
        # Vocabulary sophistication
        if features['avg_word_length'] > 5.5:
            indicators.append({
                'type': 'AI Indicator',
                'description': 'High vocabulary sophistication',
                'value': f"Average word length: {features['avg_word_length']:.1f} chars",
                'confidence': 'Medium',
                'reasoning': 'AI tends to use longer, more formal words'
            })
        
        # Sentence consistency
        if features['sentence_length_std'] < 3:
            indicators.append({
                'type': 'AI Indicator',
                'description': 'Very consistent sentence lengths',
                'value': f"Standard deviation: {features['sentence_length_std']:.1f}",
                'confidence': 'High',
                'reasoning': 'AI generates more consistent sentence structures'
            })
        
        # Complex word usage
        if features['complex_word_ratio'] > 0.15:
            indicators.append({
                'type': 'AI Indicator',
                'description': 'High complex word usage',
                'value': f"Complex words: {features['complex_word_ratio']:.1%}",
                'confidence': 'Medium',
                'reasoning': 'AI often uses more complex vocabulary'
            })
        
        # Perfect grammar (low punctuation errors)
        if features['punctuation_ratio'] > 0.08:
            indicators.append({
                'type': 'AI Indicator',
                'description': 'High punctuation density',
                'value': f"Punctuation ratio: {features['punctuation_ratio']:.1%}",
                'confidence': 'Low',
                'reasoning': 'Very formal punctuation usage'
            })
        
        # Human indicators
        if features['sentence_length_std'] > 8:
            indicators.append({
                'type': 'Human Indicator',
                'description': 'Variable sentence lengths',
                'value': f"Standard deviation: {features['sentence_length_std']:.1f}",
                'confidence': 'High',
                'reasoning': 'Humans write with more natural variation'
            })
        
        if features['lexical_diversity'] < 0.7:
            indicators.append({
                'type': 'Human Indicator',
                'description': 'Lower lexical diversity',
                'value': f"Unique words: {features['lexical_diversity']:.1%}",
                'confidence': 'Medium',
                'reasoning': 'Humans tend to repeat words more naturally'
            })
        
        return indicators
    
    def create_dual_model_comparison_chart(self, predictions):
        """Create comparison chart for both models"""
        if not all(predictions.values()):
            return None
            
        models = ['RoBERTa', 'DistilBERT']
        human_probs = [predictions[model]['human_probability'] for model in models]
        ai_probs = [predictions[model]['ai_probability'] for model in models]
        confidences = [predictions[model]['confidence'] for model in models]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Probability Comparison', 'Confidence Comparison', 
                          'Prediction Agreement', 'Uncertainty Analysis'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # Probability comparison
        fig.add_trace(
            go.Bar(name='Human', x=models, y=human_probs, marker_color='#4ECDC4'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='AI Generated', x=models, y=ai_probs, marker_color='#FF6B6B'),
            row=1, col=1
        )
        
        # Confidence comparison
        fig.add_trace(
            go.Bar(name='Confidence', x=models, y=confidences, marker_color='#45B7D1'),
            row=1, col=2
        )
        
        # Agreement indicator
        agreement = predictions['RoBERTa']['prediction'] == predictions['DistilBERT']['prediction']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=100 if agreement else 0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Model Agreement %"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "#4ECDC4" if agreement else "#FF6B6B"},
                      'steps': [{'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 100], 'color': "gray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=2, col=1
        )
        
        # Uncertainty analysis
        uncertainties = [predictions[model]['uncertainty'] for model in models]
        fig.add_trace(
            go.Bar(name='Uncertainty', x=models, y=uncertainties, marker_color='#FFA07A'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Dual Model Analysis Dashboard")
        return fig

    def create_feature_comparison_chart(self, features):
        """Create radar chart comparing features to typical AI/Human patterns"""
        
        # Typical ranges (these would be calibrated from training data)
        ai_typical = {
            'avg_word_length': 0.8,  # Normalized scores
            'lexical_diversity': 0.9,
            'complex_word_ratio': 0.8,
            'sentence_length_std': 0.3,
            'punctuation_ratio': 0.7
        }
        
        human_typical = {
            'avg_word_length': 0.6,
            'lexical_diversity': 0.6,
            'complex_word_ratio': 0.4,
            'sentence_length_std': 0.8,
            'punctuation_ratio': 0.5
        }
        
        # Normalize current features
        current_normalized = {
            'avg_word_length': min(features['avg_word_length'] / 7, 1),
            'lexical_diversity': features['lexical_diversity'],
            'complex_word_ratio': min(features['complex_word_ratio'] / 0.3, 1),
            'sentence_length_std': min(features['sentence_length_std'] / 15, 1),
            'punctuation_ratio': min(features['punctuation_ratio'] / 0.15, 1)
        }
        
        categories = list(ai_typical.keys())
        
        fig = go.Figure()
        
        # Add AI typical pattern
        fig.add_trace(go.Scatterpolar(
            r=[ai_typical[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name='Typical AI Pattern',
            line_color='red',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        # Add Human typical pattern
        fig.add_trace(go.Scatterpolar(
            r=[human_typical[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name='Typical Human Pattern',
            line_color='blue',
            fillcolor='rgba(0, 0, 255, 0.2)'
        ))
        
        # Add current text pattern
        fig.add_trace(go.Scatterpolar(
            r=[current_normalized[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name='Your Text',
            line_color='green',
            fillcolor='rgba(0, 255, 0, 0.3)',
            line_width=3
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Text Pattern Analysis"
        )
        
        return fig


def show_prediction_report_page(models, device):
    """Main function to display the dual model AI detection analysis page"""
    
    # Custom CSS for report page
    st.markdown("""
    <style>
    .report-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #2c3e50;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .model-comparison-box {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        min-height: 250px;
    }
    
    .model-comparison-box h3 {
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 0.5rem;
        border-radius: 6px;
    }
    
    .roberta-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #d32f2f;
    }
    
    .distilbert-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #1976d2;
    }
    
    .indicator-card {
        border-left: 4px solid;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .ai-indicator {
        border-left-color: #FF6B6B;
        background: linear-gradient(135deg, #32CD32 0%, #006400 100%);
    }
    
    .human-indicator {
        border-left-color: #4ECDC4;
        background: linear-gradient(135deg, #32CD32 0%, #006400 100%);
    }
    
    .agreement-indicator {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #32CD32 0%, #006400 100%);
    }
    
    .disagreement-indicator {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #dc3545 100%);
    }
    
    .step-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #dee2e6 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    .intro-box {
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .step-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="report-header">🔄 Dual Model AI Detection Analysis</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="intro-box">
    🤖 <strong>Compare RoBERTa vs DistilBERT</strong><br>
    See how both models analyze the same text and compare their predictions, confidence levels, and decision patterns.
    </div>
    """, unsafe_allow_html=True)
    
    # Get text input for analysis
    st.header("📝 Text Analysis Input")
    
    # Check if text was passed from main page
    if 'analysis_text' in st.session_state and st.session_state.analysis_text:
        text_to_analyze = st.session_state.analysis_text
        
        st.text_area("Text being analyzed:", text_to_analyze, height=100, disabled=True)
        
        # Clear the session state
        if st.button("🔄 Analyze Different Text"):
            del st.session_state.analysis_text
            st.rerun()
    else:
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Use Sample Text"],
            horizontal=True
        )
        
        text_to_analyze = ""
        
        if input_method == "Type/Paste Text":
            text_to_analyze = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Enter text for dual model analysis...",
                help="Enter at least 50 words for comprehensive comparison"
            )
        else:
            sample_type = st.selectbox(
                "Choose sample:",
                ["AI-Generated Sample", "Human-Written Sample"]
            )
            
            if sample_type == "AI-Generated Sample":
                text_to_analyze = """The implementation of artificial intelligence in contemporary educational systems represents a paradigmatic shift toward enhanced pedagogical methodologies. Modern educational institutions are increasingly integrating sophisticated AI technologies to optimize learning outcomes and facilitate personalized educational experiences. These technological innovations enable comprehensive assessment of individual learning patterns, thereby allowing educators to tailor instructional approaches to meet specific student requirements."""
            else:
                text_to_analyze = """i think AI is pretty cool but sometimes it makes weird mistakes. like yesterday when i was using chatgpt it told me that penguins could fly which is obviously wrong lol. but overall its still helpful for homework and stuff. my teacher said we shouldnt use it too much though because we need to learn things ourselves. i guess thats true but its tempting when you have a hard assignment due tomorrow!"""
            
            st.text_area("Sample text:", text_to_analyze, height=100, disabled=True)
    
    # Check if both models are available
    available_models = [name for name in models.keys() if models[name]['loaded']]
    
    if len(available_models) < 2:
        st.error("❌ Both models (RoBERTa and DistilBERT) need to be available for comparison analysis.")
        st.info("Please train both models using the Jupyter notebooks before using this analysis.")
        return

    if text_to_analyze and len(text_to_analyze.strip()) > 20:
        # Initialize dual model report generator
        report_generator = DualModelAIDetectionReport(models, device)
        
        if st.button("🔄 Generate Dual Model Analysis", type="primary"):
            with st.spinner("Analyzing text with both models..."):
                
                # Add spacing
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Step 1: Get predictions from both models
                st.markdown('<div class="step-header">⚖️ Step 1: Dual Model Predictions</div>', unsafe_allow_html=True)
                
                predictions = report_generator.get_dual_model_predictions(text_to_analyze)
                
                # Display side-by-side predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    if predictions['RoBERTa']:
                        pred_text = "AI Generated" if predictions['RoBERTa']['prediction'] == 1 else "Human Written"
                        confidence = predictions['RoBERTa']['confidence']
                        
                        st.markdown('<h3 class="roberta-header">🔴 RoBERTa Analysis</h3>', unsafe_allow_html=True)
                        
                        st.metric("Prediction", pred_text)
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.metric("Human Probability", f"{predictions['RoBERTa']['human_probability']:.1%}")
                        st.metric("AI Probability", f"{predictions['RoBERTa']['ai_probability']:.1%}")
                    else:
                        st.markdown("""
                        <div class="model-comparison-box">
                        <h3 class="roberta-header">� RoBERTa Analysis</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        st.error("RoBERTa model not available")
                
                with col2:
                    if predictions['DistilBERT']:
                        pred_text = "AI Generated" if predictions['DistilBERT']['prediction'] == 1 else "Human Written"
                        confidence = predictions['DistilBERT']['confidence']
                        
                        st.markdown('<h3 class="distilbert-header">🔵 DistilBERT Analysis</h3>', unsafe_allow_html=True)
                        
                        st.metric("Prediction", pred_text)
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.metric("Human Probability", f"{predictions['DistilBERT']['human_probability']:.1%}")
                        st.metric("AI Probability", f"{predictions['DistilBERT']['ai_probability']:.1%}")
                    else:
                        st.markdown('<h3 class="distilbert-header">🔵 DistilBERT Analysis</h3>', unsafe_allow_html=True)
                        st.error("DistilBERT model not available")
                
                # Model Agreement Analysis
                if all(predictions.values()):
                    # Add spacing
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="step-header">🤝 Step 2: Model Agreement Analysis</div>', unsafe_allow_html=True)
                    
                    agreement_analysis = report_generator.analyze_model_agreement(predictions)
                    
                    # Agreement status
                    agreement_class = "agreement-indicator" if agreement_analysis['agreement'] else "disagreement-indicator"
                    agreement_text = "✅ AGREE" if agreement_analysis['agreement'] else "❌ DISAGREE"
                    
                    st.markdown(f"""
                    <div class="indicator-card {agreement_class}">
                    <strong>Model Agreement: {agreement_text}</strong><br>
                    📊 Confidence Difference: {agreement_analysis['confidence_difference']:.1%}<br>
                    🎯 Consensus Strength: {agreement_analysis['consensus_strength']}<br>
                    💡 Both models {'reached the same conclusion' if agreement_analysis['agreement'] else 'have different predictions'}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Dual model comparison dashboard
                    comparison_chart = report_generator.create_dual_model_comparison_chart(predictions)
                    if comparison_chart:
                        st.plotly_chart(comparison_chart, use_container_width=True)
                
                # Step 3: Feature Analysis
                # Add spacing
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="step-header">📊 Step 3: Linguistic Feature Analysis</div>', unsafe_allow_html=True)
                
                features = report_generator.analyze_text_features(text_to_analyze)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("📈 Text Statistics")
                    st.metric("Word Count", features['word_count'])
                    st.metric("Sentence Count", features['sentence_count'])
                    st.metric("Avg Words/Sentence", f"{features['avg_words_per_sentence']:.1f}")
                    
                with col2:
                    st.subheader("🔤 Language Features")
                    st.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")
                    st.metric("Lexical Diversity", f"{features['lexical_diversity']:.2f}")
                    st.metric("Complex Word Ratio", f"{features['complex_word_ratio']:.1%}")
                
                with col3:
                    st.subheader("📝 Style Analysis")
                    st.metric("Sentence Variation", f"{features['sentence_length_std']:.1f}")
                    st.metric("Punctuation Ratio", f"{features['punctuation_ratio']:.1%}")
                    st.metric("Uppercase Ratio", f"{features['uppercase_ratio']:.1%}")
                
                # Pattern Analysis
                indicators = report_generator.generate_ai_indicators(text_to_analyze, features)
                
                if indicators:
                    st.subheader("🔍 Pattern Detection Indicators")
                    
                    for indicator in indicators:
                        indicator_class = "ai-indicator" if indicator['type'] == 'AI Indicator' else "human-indicator"
                        
                        st.markdown(f"""
                        <div class="indicator-card {indicator_class}">
                        <strong>{indicator['type']}: {indicator['description']}</strong><br>
                        📊 {indicator['value']}<br>
                        🎯 Confidence: {indicator['confidence']}<br>
                        💡 {indicator['reasoning']}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Feature comparison radar chart
                st.subheader("📈 Text Pattern Comparison")
                radar_chart = report_generator.create_feature_comparison_chart(features)
                st.plotly_chart(radar_chart, use_container_width=True)
                
                # Step 4: Final Consensus
                if all(predictions.values()):
                    st.header("🎯 Step 4: Final Consensus Analysis")
                    
                    roberta_pred = predictions['RoBERTa']['prediction']
                    distilbert_pred = predictions['DistilBERT']['prediction']
                    
                    if roberta_pred == distilbert_pred:
                        consensus_text = "AI Generated" if roberta_pred == 1 else "Human Written"
                        avg_confidence = (predictions['RoBERTa']['confidence'] + predictions['DistilBERT']['confidence']) / 2
                        verdict_color = "#FF6B6B" if roberta_pred == 1 else "#4ECDC4"
                        icon = "🤖" if roberta_pred == 1 else "👤"
                        
                        st.markdown(f"""
                        <div style="
                            border: 3px solid {verdict_color};
                            border-radius: 15px;
                            padding: 2rem;
                            text-align: center;
                            background-color: {'#FFE5E5' if roberta_pred == 1 else '#E5F9F7'};
                            margin: 2rem 0;
                        ">
                        <h2 style="color: {verdict_color}; margin: 0;">
                        ✅ CONSENSUS: {icon} {consensus_text}
                        </h2>
                        <h3 style="margin: 0.5rem 0; color: {verdict_color}">Average Confidence: {avg_confidence:.1%}</h3>
                        <p style="margin: 0.5rem 0; color: {verdict_color}">Both models agree on this prediction</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="
                            border: 3px solid #FFA500;
                            border-radius: 15px;
                            padding: 2rem;
                            text-align: center;
                            background-color: #FFF8DC;
                            margin: 2rem 0;
                        ">
                        <h2 style="color: #FF8C00; margin: 0;">
                        ⚠️ DISAGREEMENT: Models Have Different Predictions
                        </h2>
                        <p style="margin: 0.5rem 0; color: #FF8C00;">
                        RoBERTa: {'🤖 AI Generated' if roberta_pred == 1 else '👤 Human Written'} | 
                        DistilBERT: {'🤖 AI Generated' if distilbert_pred == 1 else '👤 Human Written'}
                        </p>
                        <p style="margin: 0.5rem 0; color: #FF8C00;">
                        Consider the confidence levels and features above for final decision
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Export comprehensive report
                st.header("📥 Export Comprehensive Report")
                
                if all(predictions.values()):
                    report_data = {
                        'text_analyzed': text_to_analyze,
                        'roberta_prediction': predictions['RoBERTa'],
                        'distilbert_prediction': predictions['DistilBERT'],
                        'model_agreement': agreement_analysis,
                        'features': features,
                        'indicators': indicators,
                        'consensus': roberta_pred == distilbert_pred
                    }
                    
                    st.download_button(
                        label="📊 Download Dual Model Analysis (JSON)",
                        data=json.dumps(report_data, indent=2),
                        file_name="dual_model_ai_detection_report.json",
                        mime="application/json"
                    )
    
    elif text_to_analyze:
        st.warning("Please enter at least 20 characters for analysis.")
    
    else:
        st.info("👆 Enter text above to see comprehensive dual model AI detection analysis")
    
    # Back to home button
    if st.button("🏠 Back to Quick Detection"):
        st.session_state.page = "home"
        st.rerun()
