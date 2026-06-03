# AutoComplete - Language Modeling Project

A comprehensive language modeling project that trains a neural network on the WikiText-103 dataset and provides an interactive web-based auto-complete application.

## Project Overview

This project consists of two main components:

1. **Training Pipeline (`main.py`)**: Trains a custom Simple Language Model (SLM) on the WikiText-103 dataset for next-token prediction.
2. **Web Application (`application.py`)**: A Flask-based interactive web application that uses the trained model to provide real-time text auto-completion suggestions.

The model uses BERT-base-uncased tokenizer and processes text with sequence length of 70 tokens. The architecture includes embedding layers, multi-layer RNN with dropout regularization, and is trained using optimization techniques like gradient clipping and weight decay.

## Project Structure

```
.
├── main.py                 # Training pipeline entry point
├── application.py          # Flask web application entry point
├── model_zip.py           # Model packaging utility
├── requirements.txt       # Python dependencies
├── config/
│   └── config.yaml        # Configuration for training and model
├── src/
│   ├── model.py           # Language model architecture
│   ├── train.py           # Training loop logic
│   ├── evaluation.py      # Model evaluation metrics
│   ├── generate.py        # Text generation utilities
│   ├── data_ingestion.py  # Dataset download and loading
│   └── data_processing.py # Tokenization and data preparation
├── utils/
│   ├── config_manager.py  # Configuration loader
│   ├── logger.py          # Logging setup
│   └── common_functions.py # Shared utility functions
├── templates/
│   └── index.html         # Web app UI template
├── static/
│   ├── styles.css         # Web app styling
│   └── script.js          # Web app JavaScript
├── artifacts/             # Trained model weights
├── cache/                 # Cached tokenized datasets
└── data/
    └── raw/              # Raw dataset storage
```

## Installation

### Prerequisites
- Python 3.7+
- CUDA 11.0+ (for GPU support, optional)
- pip

### Setup Steps

1. **Clone/Navigate to the project directory:**
   ```bash
   cd AutoComplete
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config/config.yaml` to customize:
- **Model Architecture**: `embedding_dim`, `hidden_dim`, `num_layers`, dropout rates
- **Training Parameters**: `num_epochs`, `batch_size`, `learning_rate`, `clip`, `weight_decay`
- **Data Settings**: `seq_len` (sequence length), tokenizer name
- **Paths**: Output directories for logs, models, and results

## How to Run

### 1. Training Pipeline

To train the language model on WikiText-103:

```bash
python main.py
```

This will:
- Download the WikiText-103 dataset (cached for reuse)
- Tokenize data using BERT-base-uncased tokenizer
- Create train/validation/test data loaders
- Train the model for the specified number of epochs
- Evaluate on validation and test sets
- Save the best model to `artifacts/best_model.pt`
- Generate training logs and metrics

**Training Output:**
- Logs: `./logs/training.log`
- Best Model: `./artifacts/best_model.pt`
- Metrics: `./results/metrics/`
- Plots: `./results/plots/`

### 2. Web Application

To run the interactive auto-complete web application:

```bash
python application.py
```

This will:
- Load the trained model from `artifacts/best_model.pt`
- Start a Flask development server on `http://localhost:5000`
- Open the web interface in your browser

**Using the Web App:**
1. Navigate to `http://localhost:5000` in your browser
2. Type or paste text in the input box
3. The application will suggest the next word completion
4. Select a suggestion to auto-complete or continue typing

**Features:**
- Real-time token suggestions
- Automatic filtering of invalid tokens (punctuation, special tokens)
- Clean token processing (removes BERT prefixes like `##` and `Ġ`)
- GPU acceleration (if available)

## Model Architecture

The Simple Language Model (SLM) consists of:
- **Embedding Layer**: Converts token IDs to embeddings (default: 300 dimensions)
- **RNN Layers**: Multi-layer LSTM/GRU (default: 3 layers, 1150 hidden units)
- **Dropout**: Applied at multiple stages for regularization
  - `dropouti`: Input dropout (0.65)
  - `dropoute`: Embedding dropout (0.1)
  - `dropouth`: Hidden state dropout (0.3)
  - `dropouto`: Output dropout (0.4)
- **Output Layer**: Linear projection to vocabulary size

**Configuration in `config/config.yaml`:**
```yaml
model:
  architecture:
    embedding_dim: 300
    num_layers: 3
    hidden_dim: 1150
    dropoute: 0.1
    dropouti: 0.65
    dropouth: 0.3
    dropouto: 0.4
```

## Training Details

- **Dataset**: WikiText-103-raw-v1 (103 million tokens)
- **Tokenizer**: BERT-base-uncased
- **Sequence Length**: 70 tokens
- **Batch Size**: 32
- **Learning Rate**: 7.5 (with scheduling)
- **Optimizer**: Momentum (momentum=0.9)
- **Regularization**: 
  - Weight decay: 1.2e-6
  - Gradient clipping: 0.25
- **Early Stopping**: Patience of 5 epochs
- **Epochs**: 50 (with early stopping)

## Requirements

The project requires the following Python packages (see `requirements.txt`):
- `torch`: Deep learning framework
- `transformers`: Pre-trained models and tokenizers
- `datasets`: Dataset loading and processing
- `flask`: Web framework
- `torchmetrics`: Evaluation metrics

## Troubleshooting

### GPU Not Found
If training/application doesn't use GPU:
1. Ensure CUDA is installed: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Out of Memory
- Reduce `batch_size` in `config/config.yaml`
- Reduce `seq_len` for shorter sequences
- Reduce `hidden_dim` or use fewer `num_layers`

### Model Not Found
- Ensure `artifacts/best_model.pt` exists before running the web app
- Run training pipeline first: `python main.py`

## License

This project is provided as-is for educational and research purposes.

## Author Notes

- Training takes significant time depending on hardware (GPU recommended)
- First run will download WikiText-103 dataset (~4GB)
- Cached tokenized data is stored in `cache/` for faster subsequent runs
- The web application uses the best model from training
