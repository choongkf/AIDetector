# AI Detector Project

This project implements an AI text detector using machine learning models to distinguish between human-written and AI-generated text.

## Features

- **Two Model Comparison**: Compare RoBERTa and DistilBERT models for AI detection
- **Jupyter Notebook Training**: Interactive model training and evaluation
- **Streamlit Web Interface**: User-friendly web app for real-time AI detection
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations

## Project Structure

```
AIDetector/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training_roberta.ipynb
│   ├── 03_model_training_distilbert.ipynb
│   └── 04_model_comparison.ipynb
├── streamlit_app/
│   ├── app.py
│   └── utils.py
├── src/
│   ├── data_loader.py
│   ├── model_trainer.py
│   └── predictor.py
├── models/
├── data/
└── requirements.txt
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training Notebooks**:
   - Open Jupyter notebooks in the `notebooks/` folder
   - Run them in order (01 → 02 → 03 → 04)

3. **Launch Streamlit App**:
   ```bash
   streamlit run streamlit_app/app.py
   ```

## Suggested Models

### 1. RoBERTa (Primary Model)
- **Model**: `roberta-base`
- **Strengths**: Excellent text understanding, robust performance
- **Use Case**: High-accuracy AI detection

### 2. DistilBERT (Lightweight Model)
- **Model**: `distilbert-base-uncased`
- **Strengths**: Faster inference, smaller size
- **Use Case**: Real-time applications with speed requirements

## Dataset Recommendations

For training, consider these datasets:
- **GPT-2 Output Dataset**: Contains human vs AI-generated text
- **HC3 Dataset**: Human vs ChatGPT comparisons
- **Custom Dataset**: Mix of human writing and outputs from various AI models

## Usage

1. **Training**: Use Jupyter notebooks to train and evaluate models
2. **Inference**: Use the Streamlit app to test text samples
3. **Comparison**: Compare model performance using the evaluation notebook

## Model Performance Metrics

The project evaluates models using:
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Speed benchmarks

## Contributing

1. Add new models in the `src/` directory
2. Create corresponding training notebooks
3. Update the Streamlit app for new model integration
4. Add comprehensive evaluation metrics
