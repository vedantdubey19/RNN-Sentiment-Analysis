# 🎭 RNN Sentiment Analysis

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Enabled-red.svg?logo=keras&logoColor=white)](https://keras.io/)
[![Natural Language Processing](https://img.shields.io/badge/Domain-NLP-green.svg)]()

A recurrent neural network project demonstrating text sentiment classification (positive/negative) using Keras and TensorFlow. The project implements and compares two architectures:
1. **Simple RNN WITHOUT Embedding** (uses raw index sequences)
2. **Simple RNN WITH Embedding** (learns continuous word representations)

---

## ⚙️ Process Flowchart

Here is the step-by-step pipeline representing data processing, tokenization, model training, and prediction:

```mermaid
flowchart TD
    A[Text Sentences] --> B[Tokenizer]
    B -->|fit_on_texts| C[Word Vocabulary]
    C -->|texts_to_sequences| D[Numeric Sequences]
    D -->|pad_sequences| E[Padded Input Sequences]
    E --> F{Model Configuration}
    F -->|Without Embedding| G[Reshape to 3D & SimpleRNN]
    F -->|With Embedding| H[Embedding Layer + SimpleRNN]
    G --> I[Dense Classifier Sigmoid]
    H --> I
    I -->|Binary Sentiment| J[Sentiment Prediction: Positive / Negative]
```

---

## 🧠 Model Architectures

### 1. RNN WITHOUT Embedding
- **Input:** Padded sequences reshaped into 3D tensors `(batch_size, sequence_length, 1)`.
- **SimpleRNN Layer:** 10 recurrent units processing sequences step-by-step.
- **Dense Layer:** 1 unit with Sigmoid activation for binary classification.

### 2. RNN WITH Embedding
- **Input:** Padded sequences of shape `(batch_size, sequence_length)`.
- **Embedding Layer:** Projects discrete token indices to an 8-dimensional continuous vector space.
- **SimpleRNN Layer:** 10 recurrent units processing the embedding vectors.
- **Dense Layer:** 1 unit with Sigmoid activation.

---

## 📈 Training Performance

- **Dataset:** Small illustrative corpus of positive and negative sentiment sentences.
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Performance:** Both models easily learn the training sentences perfectly (100% accuracy) and generalization differences are explored using out-of-distribution test sentences.

---

## 🛠️ Setup & Usage

### 1. Install Dependencies
Make sure Python is installed, then install the required dependencies:
```bash
pip install tensorflow numpy
```

### 2. Run the Notebook
Open the notebook and execute the cells:
```bash
jupyter notebook RNN_SentimentAnalysis.ipynb
```
