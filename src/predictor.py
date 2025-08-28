import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import time
import json
from typing import Tuple, Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

class AITextPredictor:
    """AI Text Detection Predictor class"""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """
        Initialize the predictor
        
        Args:
            model_path (str): Path to the saved model
            device (torch.device): Device to use for inference
        """
        self.model_path = Path(model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.model_info = {}
        
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model and tokenizer
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.model_path.exists():
                print(f"Model path does not exist: {self.model_path}")
                return False
            
            print(f"Loading model from {self.model_path}...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                num_labels=2
            )
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Store model info
            self.model_info = {
                'model_path': str(self.model_path),
                'device': str(self.device),
                'parameters': self.model.num_parameters(),
                'architecture': self.model.__class__.__name__
            }
            
            print(f"Model loaded successfully!")
            print(f"Parameters: {self.model_info['parameters']:,}")
            print(f"Device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Predict if text is AI-generated or human-written
        
        Args:
            text (str): Text to analyze
            max_length (int): Maximum sequence length
        
        Returns:
            dict: Prediction results
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Cannot make predictions.")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction with timing
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Extract results
        pred_label = prediction.item()
        probs = probabilities[0].cpu().numpy()
        
        result = {
            'prediction': 'AI Generated' if pred_label == 1 else 'Human Written',
            'prediction_label': pred_label,
            'confidence': float(probs[pred_label]),
            'probabilities': {
                'human': float(probs[0]),
                'ai_generated': float(probs[1])
            },
            'inference_time_ms': inference_time,
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 16, 
                     max_length: int = 512) -> List[Dict[str, Any]]:
        """
        Predict for multiple texts
        
        Args:
            texts (List[str]): List of texts to analyze
            batch_size (int): Batch size for processing
            max_length (int): Maximum sequence length
        
        Returns:
            List[Dict]: List of prediction results
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Cannot make predictions.")
        
        results = []
        total_time = 0
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict batch
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
            end_time = time.time()
            
            batch_time = (end_time - start_time) * 1000
            total_time += batch_time
            
            # Process results
            batch_predictions = predictions.cpu().numpy()
            batch_probabilities = probabilities.cpu().numpy()
            
            for j, (text, pred, probs) in enumerate(zip(batch_texts, batch_predictions, batch_probabilities)):
                result = {
                    'prediction': 'AI Generated' if pred == 1 else 'Human Written',
                    'prediction_label': int(pred),
                    'confidence': float(probs[pred]),
                    'probabilities': {
                        'human': float(probs[0]),
                        'ai_generated': float(probs[1])
                    },
                    'inference_time_ms': batch_time / len(batch_texts),
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'batch_index': i + j
                }
                results.append(result)
        
        # Add overall statistics
        avg_time = total_time / len(texts)
        for result in results:
            result['avg_inference_time_ms'] = avg_time
            result['total_processing_time_ms'] = total_time
        
        return results
    
    def analyze_text_features(self, text: str) -> Dict[str, Any]:
        """
        Analyze linguistic features of the text
        
        Args:
            text (str): Text to analyze
        
        Returns:
            dict: Text features
        """
        import re
        
        features = {
            # Basic counts
            'character_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            
            # Punctuation analysis
            'punctuation_count': len(re.findall(r'[.!?;,:()]', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'comma_count': text.count(','),
            
            # Character analysis
            'uppercase_count': sum(1 for c in text if c.isupper()),
            'lowercase_count': sum(1 for c in text if c.islower()),
            'digit_count': sum(1 for c in text if c.isdigit()),
            'space_count': text.count(' '),
        }
        
        # Ratios
        char_count = features['character_count']
        if char_count > 0:
            features['punctuation_ratio'] = features['punctuation_count'] / char_count
            features['uppercase_ratio'] = features['uppercase_count'] / char_count
            features['digit_ratio'] = features['digit_count'] / char_count
        else:
            features['punctuation_ratio'] = 0
            features['uppercase_ratio'] = 0
            features['digit_ratio'] = 0
        
        # Word-based features
        words = text.split()
        if words:
            word_lengths = [len(word.strip('.,!?;:()')) for word in words]
            features['avg_word_length'] = np.mean(word_lengths)
            features['max_word_length'] = max(word_lengths)
            features['min_word_length'] = min(word_lengths)
            features['unique_word_ratio'] = len(set(words)) / len(words)
            features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        else:
            features.update({
                'avg_word_length': 0,
                'max_word_length': 0,
                'min_word_length': 0,
                'unique_word_ratio': 0,
                'avg_sentence_length': 0
            })
        
        return features
    
    def detailed_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform detailed analysis including prediction and text features
        
        Args:
            text (str): Text to analyze
        
        Returns:
            dict: Complete analysis results
        """
        # Get prediction
        prediction_result = self.predict(text)
        
        # Get text features
        text_features = self.analyze_text_features(text)
        
        # Combine results
        analysis = {
            'prediction_result': prediction_result,
            'text_features': text_features,
            'model_info': self.model_info,
            'analysis_timestamp': time.time()
        }
        
        return analysis
    
    def benchmark_speed(self, sample_texts: List[str], num_runs: int = 5) -> Dict[str, float]:
        """
        Benchmark prediction speed
        
        Args:
            sample_texts (List[str]): Sample texts for benchmarking
            num_runs (int): Number of benchmark runs
        
        Returns:
            dict: Benchmark results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot benchmark.")
        
        print(f"Benchmarking with {len(sample_texts)} samples over {num_runs} runs...")
        
        times = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            for text in sample_texts:
                self.predict(text)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_total_time = np.mean(times)
        std_total_time = np.std(times)
        avg_time_per_sample = avg_total_time / len(sample_texts)
        samples_per_second = len(sample_texts) / avg_total_time
        
        results = {
            'avg_total_time': avg_total_time,
            'std_total_time': std_total_time,
            'avg_time_per_sample': avg_time_per_sample,
            'avg_time_per_sample_ms': avg_time_per_sample * 1000,
            'samples_per_second': samples_per_second,
            'num_samples': len(sample_texts),
            'num_runs': num_runs
        }
        
        print(f"Benchmark completed!")
        print(f"Average time per sample: {results['avg_time_per_sample_ms']:.1f}ms")
        print(f"Samples per second: {results['samples_per_second']:.1f}")
        
        return results
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info.copy()

def load_predictor(model_path: str, device: Optional[torch.device] = None) -> Optional[AITextPredictor]:
    """
    Convenience function to load a predictor
    
    Args:
        model_path (str): Path to saved model
        device (torch.device): Device to use
    
    Returns:
        AITextPredictor: Loaded predictor or None if failed
    """
    try:
        predictor = AITextPredictor(model_path, device)
        if predictor.is_model_loaded():
            return predictor
        else:
            return None
    except Exception as e:
        print(f"Failed to load predictor: {e}")
        return None
