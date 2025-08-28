import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Trainer class for transformer-based AI detection models"""
    
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        """
        Initialize the model trainer
        
        Args:
            model_name (str): Name of the pre-trained model (e.g., 'roberta-base', 'distilbert-base-uncased')
            device (torch.device): Device to use for training
        """
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_args = None
        
        print(f"Initializing trainer for {model_name}")
        print(f"Using device: {self.device}")
    
    def load_model(self) -> None:
        """Load the pre-trained model and tokenizer"""
        try:
            print(f"Loading {self.model_name}...")            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False
            )
            
            print(f"Model loaded successfully!")
            print(f"Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def setup_training_args(self, output_dir: str, **kwargs) -> None:
        """
        Setup training arguments
        
        Args:
            output_dir (str): Directory to save model outputs
            **kwargs: Additional training arguments
        """
        default_args = {
            'output_dir': output_dir,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 16,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'logging_dir': f"{output_dir}/logs",
            'logging_steps': 100,
            'evaluation_strategy': "steps",
            'eval_steps': 500,
            'save_steps': 1000,
            'save_total_limit': 2,
            'load_best_model_at_end': True,
            'metric_for_best_model': "accuracy",
            'greater_is_better': True,
            'push_to_hub': False,
            'dataloader_num_workers': 0,  # Windows compatibility
            'fp16': torch.cuda.is_available(),
            'learning_rate': 2e-5,
        }
        
        # Update with provided kwargs
        default_args.update(kwargs)
        
        self.training_args = TrainingArguments(**default_args)
        print(f"Training arguments configured.")
        print(f"Batch size: {self.training_args.per_device_train_batch_size}")
        print(f"Epochs: {self.training_args.num_train_epochs}")
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Args:
            eval_pred (EvalPrediction): Evaluation predictions
        
        Returns:
            dict: Computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_dataset, eval_dataset) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        
        Returns:
            dict: Training results
        """
        if self.model is None or self.training_args is None:
            raise ValueError("Model and training arguments must be set before training")
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        print("Starting training...")
        start_time = time.time()
        
        # Train the model
        training_result = self.trainer.train()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print("Training completed!")
        print(f"Training loss: {training_result.training_loss:.4f}")
        print(f"Training time: {training_time/60:.1f} minutes")
        
        return {
            'training_loss': training_result.training_loss,
            'training_time': training_time,
            'global_step': training_result.global_step
        }
    
    def evaluate(self, test_dataset) -> Dict[str, Any]:
        """
        Evaluate the model on test data
        
        Args:
            test_dataset: Test dataset
        
        Returns:
            dict: Evaluation results
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating on test set...")
        start_time = time.time()
        
        test_results = self.trainer.evaluate(test_dataset)
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        print("Evaluation completed!")
        
        # Get predictions for detailed analysis
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        results = {
            'test_accuracy': test_results['eval_accuracy'],
            'test_f1': test_results['eval_f1'],
            'test_precision': test_results['eval_precision'],
            'test_recall': test_results['eval_recall'],
            'confusion_matrix': cm.tolist(),
            'specificity': specificity,
            'sensitivity': sensitivity,
            'eval_time': eval_time,
            'samples_per_second': len(test_dataset) / eval_time
        }
        
        return results
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model and tokenizer
        
        Args:
            save_path (str): Path to save the model
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Model saved to: {save_path}")
    
    def save_results(self, results: Dict[str, Any], save_path: str) -> None:
        """
        Save training and evaluation results
        
        Args:
            results (dict): Results to save
            save_path (str): Path to save results
        """
        # Add model info
        results_with_info = {
            'model_name': self.model_name,
            'model_parameters': self.model.num_parameters() if self.model else None,
            'device': str(self.device),
            **results
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_with_info, f, indent=2)
        
        print(f"Results saved to: {save_path}")
    
    def benchmark_inference(self, test_texts: list, num_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed
        
        Args:
            test_texts (list): List of test texts
            num_samples (int): Number of samples to benchmark
        
        Returns:
            dict: Benchmark results
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be loaded before benchmarking")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Select random samples for benchmarking
        sample_texts = test_texts[:num_samples]
        
        print(f"Benchmarking inference speed with {num_samples} samples...")
        
        start_time = time.time()
        
        with torch.no_grad():
            for text in sample_texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Predict
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results = {
            'total_time': total_time,
            'avg_time_per_sample': total_time / num_samples,
            'samples_per_second': num_samples / total_time,
            'inference_time_ms': (total_time / num_samples) * 1000
        }
        
        print(f"Benchmark completed!")
        print(f"Average time per sample: {results['inference_time_ms']:.1f}ms")
        print(f"Samples per second: {results['samples_per_second']:.1f}")
        
        return results
    
    def predict_single(self, text: str) -> Tuple[int, np.ndarray, float]:
        """
        Predict a single text sample
        
        Args:
            text (str): Text to predict
        
        Returns:
            tuple: (prediction, probabilities, inference_time_ms)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be loaded before prediction")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict with timing
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000
        
        return prediction.item(), probabilities[0].cpu().numpy(), inference_time

def train_model(model_name: str, train_dataset, eval_dataset, test_dataset,
                output_dir: str, **training_kwargs) -> Dict[str, Any]:
    """
    Complete training pipeline for a model
    
    Args:
        model_name (str): Name of the pre-trained model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset  
        test_dataset: Test dataset
        output_dir (str): Output directory for saving
        **training_kwargs: Additional training arguments
    
    Returns:
        dict: Complete results including training and evaluation
    """
    # Initialize trainer
    trainer = ModelTrainer(model_name)
    
    # Load model
    trainer.load_model()
    
    # Setup training
    trainer.setup_training_args(output_dir, **training_kwargs)
    
    # Train
    training_results = trainer.train(train_dataset, eval_dataset)
    
    # Evaluate
    eval_results = trainer.evaluate(test_dataset)
    
    # Benchmark
    test_texts = [test_dataset[i]['input_ids'] for i in range(min(100, len(test_dataset)))]
    benchmark_results = trainer.benchmark_inference(test_texts)
    
    # Save model
    model_save_path = f"{output_dir}/model"
    trainer.save_model(model_save_path)
    
    # Combine all results
    all_results = {
        **training_results,
        **eval_results,
        **benchmark_results
    }
    
    # Save results
    results_save_path = f"{output_dir}/results.json"
    trainer.save_results(all_results, results_save_path)
    
    return all_results
