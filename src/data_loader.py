import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    """Custom dataset class for text classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize the dataset
        
        Args:
            texts (List[str]): List of text samples
            labels (List[int]): List of labels (0=Human, 1=AI)
            tokenizer: Hugging Face tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataLoader:
    """Data loading and preprocessing utilities"""
    
    def __init__(self, csv_path: str):
        """
        Initialize the data loader
        
        Args:
            csv_path (str): Path to the CSV file containing the data
        """
        self.csv_path = csv_path
        self.df = None
        self.texts = None
        self.labels = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self) -> Tuple[List[str], List[int]]:
        """
        Preprocess the loaded data
        
        Returns:
            tuple: (texts, labels)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Extract texts and labels
        self.texts = self.df['text'].tolist()
        self.labels = self.df['generated'].tolist()
        
        # Basic preprocessing
        self.texts = [str(text).strip() for text in self.texts]
        
        print(f"Preprocessing completed!")
        print(f"Total samples: {len(self.texts)}")
        print(f"Human samples: {self.labels.count(0)}")
        print(f"AI samples: {self.labels.count(1)}")
        
        return self.texts, self.labels
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, 
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            random_state (int): Random seed for reproducibility
        
        Returns:
            dict: Dictionary containing train, val, and test splits
        """
        if self.texts is None or self.labels is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.texts, self.labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.labels
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, 
            random_state=random_state, 
            stratify=y_temp
        )
        
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        print(f"Data split completed!")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return splits
    
    def create_datasets(self, splits: Dict[str, Any], tokenizer, 
                       max_length: int = 512) -> Dict[str, TextDataset]:
        """
        Create PyTorch datasets from splits
        
        Args:
            splits (dict): Data splits from split_data()
            tokenizer: Hugging Face tokenizer
            max_length (int): Maximum sequence length
        
        Returns:
            dict: Dictionary containing train, val, and test datasets
        """
        datasets = {
            'train': TextDataset(
                splits['X_train'], splits['y_train'], tokenizer, max_length
            ),
            'val': TextDataset(
                splits['X_val'], splits['y_val'], tokenizer, max_length
            ),
            'test': TextDataset(
                splits['X_test'], splits['y_test'], tokenizer, max_length
            )
        }
        
        print(f"Datasets created successfully!")
        return datasets
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistical information about the data
        
        Returns:
            dict: Data statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Text length statistics
        text_lengths = self.df['text'].str.len()
        word_counts = self.df['text'].str.split().str.len()
        
        stats = {
            'total_samples': len(self.df),
            'text_length_stats': {
                'mean': text_lengths.mean(),
                'median': text_lengths.median(),
                'std': text_lengths.std(),
                'min': text_lengths.min(),
                'max': text_lengths.max()
            },
            'word_count_stats': {
                'mean': word_counts.mean(),
                'median': word_counts.median(),
                'std': word_counts.std(),
                'min': word_counts.min(),
                'max': word_counts.max()
            },
            'class_distribution': self.df['generated'].value_counts().to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicate_texts': self.df['text'].duplicated().sum()
        }
        
        return stats
    
    def sample_data(self, n_samples: int = 100, stratify: bool = True) -> pd.DataFrame:
        """
        Sample a subset of the data
        
        Args:
            n_samples (int): Number of samples to select
            stratify (bool): Whether to maintain class distribution
        
        Returns:
            pd.DataFrame: Sampled data
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if stratify:
            # Sample equal numbers from each class
            n_per_class = n_samples // 2
            human_samples = self.df[self.df['generated'] == 0].sample(
                n_per_class, random_state=42
            )
            ai_samples = self.df[self.df['generated'] == 1].sample(
                n_per_class, random_state=42
            )
            sampled_df = pd.concat([human_samples, ai_samples], ignore_index=True)
        else:
            sampled_df = self.df.sample(n_samples, random_state=42)
        
        return sampled_df.reset_index(drop=True)

def create_data_loaders(datasets: Dict[str, TextDataset], 
                       batch_sizes: Dict[str, int] = None) -> Dict[str, DataLoader]:
    """
    Create PyTorch data loaders from datasets
    
    Args:
        datasets (dict): Dictionary of datasets
        batch_sizes (dict): Batch sizes for each split
    
    Returns:
        dict: Dictionary of data loaders
    """
    if batch_sizes is None:
        batch_sizes = {'train': 16, 'val': 32, 'test': 32}
    
    data_loaders = {}
    
    for split, dataset in datasets.items():
        batch_size = batch_sizes.get(split, 32)
        shuffle = (split == 'train')  # Only shuffle training data
        
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Set to 0 for Windows compatibility
        )
    
    return data_loaders

def analyze_text_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the distribution of text characteristics
    
    Args:
        df (pd.DataFrame): DataFrame containing text data
    
    Returns:
        dict: Analysis results
    """
    # Calculate text features
    df_copy = df.copy()
    df_copy['text_length'] = df_copy['text'].str.len()
    df_copy['word_count'] = df_copy['text'].str.split().str.len()
    df_copy['sentence_count'] = df_copy['text'].str.count(r'\.') + 1
    
    # Group by class
    class_analysis = {}
    
    for class_label in [0, 1]:
        class_name = 'human' if class_label == 0 else 'ai'
        class_data = df_copy[df_copy['generated'] == class_label]
        
        class_analysis[class_name] = {
            'count': len(class_data),
            'text_length': {
                'mean': class_data['text_length'].mean(),
                'std': class_data['text_length'].std(),
                'median': class_data['text_length'].median()
            },
            'word_count': {
                'mean': class_data['word_count'].mean(),
                'std': class_data['word_count'].std(),
                'median': class_data['word_count'].median()
            },
            'sentence_count': {
                'mean': class_data['sentence_count'].mean(),
                'std': class_data['sentence_count'].std(),
                'median': class_data['sentence_count'].median()
            }
        }
    
    return {
        'overall_stats': {
            'total_samples': len(df),
            'balance_ratio': len(df[df['generated'] == 1]) / len(df)
        },
        'class_analysis': class_analysis
    }
