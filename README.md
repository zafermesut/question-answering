# Question Answering Model Comparison: T5 vs BERT

This project compares the performance of T5 and BERT models fine-tuned on the Stanford Question Answering Dataset (SQuAD) for question answering tasks. The implementation is optimized for MacBook M2 with 16GB RAM.

## Overview

The project implements and evaluates two popular transformer-based models:

- T5 (Text-to-Text Transfer Transformer)
- BERT (Bidirectional Encoder Representations from Transformers)

Both models were fine-tuned on a subset of the SQuAD dataset and evaluated using various metrics to ensure fair comparison.

## Setup

### Requirements

```bash
pip install torch transformers datasets evaluate scikit-learn matplotlib sentencepiece
```

### Hardware Requirements

- Tested on MacBook M2 with 16GB RAM
- Models and training parameters are optimized for memory efficiency

## Implementation Details

### Dataset

- Using Stanford Question Answering Dataset (SQuAD)
- Training on 1% of the dataset to optimize for memory constraints
- Balanced train-test split (80-20)

### Model Configurations

- T5: t5-small model
- BERT: bert-base-uncased model
- Batch size: 8
- Learning rate: 5e-5
- Training epochs: 3

## Results

### Main Metrics

- **Exact Match**
  - T5: 74.6%
  - BERT: 65.8%
- **F1 Score**
  - T5: 84.25%
  - BERT: 75.81%

### Additional Metrics

| Metric     | T5    | BERT  |
| ---------- | ----- | ----- |
| BLEU Score | 0.510 | 0.569 |
| ROUGE-1    | 0.838 | 0.759 |
| ROUGE-2    | 0.526 | 0.455 |
| ROUGE-L    | 0.840 | 0.758 |
| ROUGE-Lsum | 0.840 | 0.758 |

### Training Metrics

| Metric               | T5     | BERT   |
| -------------------- | ------ | ------ |
| Training Loss        | 0.7061 | 1.7309 |
| Evaluation Loss      | 0.0127 | 1.9257 |
| Training Runtime (s) | 963.48 | 931.97 |

## Project Structure

```
bert-query-answering/
├── train/
│   ├── bert_train.ipynb     # BERT training notebook
│   └── t5_train.ipynb       # T5 training notebook
├── test/
│   └── test.ipynb          # Model testing and comparison
├── models/                 # Saved model checkpoints
├── logs/                  # Training logs
└── results/              # Evaluation results
```

## Usage

1. Training the models:

```python
# Run the training notebooks in the train/ directory
jupyter notebook train/bert_train.ipynb
jupyter notebook train/t5_train.ipynb
```

2. Testing the models:

```python
# Run the test notebook
jupyter notebook test/test.ipynb
```

## Example Usage

```python
# T5 Model
question = "What is one popular application of machine learning?"
context = "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It involves various techniques, such as supervised learning, unsupervised learning, and reinforcement learning. One of the most popular applications of machine learning is natural language processing, which includes tasks like translation, sentiment analysis, and question answering."

# Both models correctly predict: "natural language processing"
```

## Key Findings

1. T5 outperforms BERT in most metrics, particularly in exact match and F1 score
2. T5 shows better generalization with lower evaluation loss
3. Both models have similar training runtime
4. Memory optimization techniques allow successful training on 16GB RAM

## Acknowledgments

- HuggingFace Transformers library
- Stanford Question Answering Dataset (SQuAD)
- PyTorch team for M2 optimization
