import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import re
from typing import Tuple, Dict, Any

def load_model(model_path: str, device: torch.device):
    """
    Load a trained model and tokenizer from the specified path
    
    Args:
        model_path (str): Path to the saved model
        device (torch.device): Device to load the model on
    
    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

def predict_single_text(text: str, model, tokenizer, device: torch.device, 
                       max_length: int = 512) -> Tuple[int, np.ndarray, float]:
    """
    Predict if a single text is AI-generated or human-written
    
    Args:
        text (str): Text to analyze
        model: Trained model
        tokenizer: Model tokenizer
        device (torch.device): Device for inference
        max_length (int): Maximum sequence length
    
    Returns:
        tuple: (prediction, probabilities, inference_time_ms)
    """
    # Tokenize text
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
    
    # Predict with timing
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return prediction.item(), probabilities[0].cpu().numpy(), inference_time

def predict_batch(texts: list, model, tokenizer, device: torch.device, 
                 batch_size: int = 16, max_length: int = 512) -> Dict[str, Any]:
    """
    Predict for a batch of texts
    
    Args:
        texts (list): List of texts to analyze
        model: Trained model
        tokenizer: Model tokenizer
        device (torch.device): Device for inference
        batch_size (int): Batch size for processing
        max_length (int): Maximum sequence length
    
    Returns:
        dict: Results containing predictions, probabilities, and timing info
    """
    predictions = []
    probabilities = []
    total_time = 0
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Predict batch
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_predictions = torch.argmax(batch_probabilities, dim=-1)
        end_time = time.time()
        
        total_time += (end_time - start_time)
        
        # Store results
        predictions.extend(batch_predictions.cpu().numpy())
        probabilities.extend(batch_probabilities.cpu().numpy())
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'total_time': total_time,
        'avg_time_per_sample': total_time / len(texts),
        'samples_per_second': len(texts) / total_time
    }

def extract_text_features(text: str) -> Dict[str, float]:
    """
    Extract linguistic features from text
    
    Args:
        text (str): Input text
    
    Returns:
        dict: Extracted features
    """
    features = {}
    
    # Basic counts
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
    
    # Punctuation analysis
    features['punctuation_count'] = len(re.findall(r'[.!?;,:]', text))
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # Character analysis
    features['uppercase_count'] = sum(1 for c in text if c.isupper())
    features['digit_count'] = sum(1 for c in text if c.isdigit())
    
    # Ratios
    if features['char_count'] > 0:
        features['punctuation_ratio'] = features['punctuation_count'] / features['char_count']
        features['uppercase_ratio'] = features['uppercase_count'] / features['char_count']
        features['digit_ratio'] = features['digit_count'] / features['char_count']
    else:
        features['punctuation_ratio'] = 0
        features['uppercase_ratio'] = 0
        features['digit_ratio'] = 0
    
    # Word-based features
    words = text.split()
    if words:
        features['avg_word_length'] = np.mean([len(word.strip('.,!?;:')) for word in words])
        features['unique_word_ratio'] = len(set(words)) / len(words)
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
    else:
        features['avg_word_length'] = 0
        features['unique_word_ratio'] = 0
        features['avg_sentence_length'] = 0
    
    return features

def format_prediction_result(prediction: int, probabilities: np.ndarray, 
                           inference_time: float, text: str) -> Dict[str, Any]:
    """
    Format prediction results for display
    
    Args:
        prediction (int): Model prediction (0=Human, 1=AI)
        probabilities (np.ndarray): Class probabilities
        inference_time (float): Inference time in milliseconds
        text (str): Input text
    
    Returns:
        dict: Formatted results
    """
    result = {
        'prediction': 'AI Generated' if prediction == 1 else 'Human Written',
        'prediction_label': prediction,
        'confidence': float(probabilities[prediction]),
        'human_probability': float(probabilities[0]),
        'ai_probability': float(probabilities[1]),
        'inference_time_ms': inference_time,
        'text_stats': extract_text_features(text)
    }
    
    return result

def benchmark_model(model, tokenizer, device: torch.device, 
                   sample_texts: list, num_runs: int = 5) -> Dict[str, float]:
    """
    Benchmark model performance
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        device (torch.device): Device for inference
        sample_texts (list): Sample texts for benchmarking
        num_runs (int): Number of benchmark runs
    
    Returns:
        dict: Benchmark results
    """
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        
        for text in sample_texts:
            predict_single_text(text, model, tokenizer, device)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_total_time = np.mean(times)
    std_total_time = np.std(times)
    avg_time_per_sample = avg_total_time / len(sample_texts)
    samples_per_second = len(sample_texts) / avg_total_time
    
    return {
        'avg_total_time': avg_total_time,
        'std_total_time': std_total_time,
        'avg_time_per_sample': avg_time_per_sample,
        'samples_per_second': samples_per_second,
        'num_samples': len(sample_texts),
        'num_runs': num_runs
    }

def validate_text_input(text: str, min_length: int = 10, max_length: int = 50000) -> Tuple[bool, str]:
    """
    Validate text input
    
    Args:
        text (str): Input text to validate
        min_length (int): Minimum text length
        max_length (int): Maximum text length
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    if len(text) < min_length:
        return False, f"Text must be at least {min_length} characters long"
    
    if len(text) > max_length:
        return False, f"Text must be no more than {max_length} characters long"
    
    # Check if text contains only whitespace
    if not text.strip():
        return False, "Text cannot contain only whitespace"
    
    return True, "Valid"

def get_model_info(model) -> Dict[str, Any]:
    """
    Get information about the model
    
    Args:
        model: PyTorch model
    
    Returns:
        dict: Model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': model.__class__.__name__
    }
