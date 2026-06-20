# 🎭 SentimentRNN
### An Interactive Deep Learning Sandbox for Sentiment Analysis

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Enabled-lightgrey.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![NLP Sandbox](https://img.shields.io/badge/NLP-Comparative--Sandbox-blueviolet)]()

Welcome to **SentimentRNN**! This repository is a comparative learning playground designed to demystify how recurrent neural network architectures process language. By testing four model variations side-by-side, this sandbox highlights why word representations (embeddings) and memory gating mechanisms (LSTMs & GRUs) are essential for modern NLP tasks.

---

## ⚙️ Process Flowchart

See how custom review text flows through tokenizer mappings and sequence transformations to generate real-time comparative outputs:

```mermaid
flowchart TD
    A["✍️ User Text Input"] --> B["✂️ Preprocessor & Tokenizer"]
    B -->|fit_on_texts| C["📚 Vocabulary Index"]
    C -->|texts_to_sequences| D["🔢 Index Sequence"]
    D -->|pad_sequences| E["🛡️ Pre-Padded Sequence (max_len=7)"]
    E --> F{"🤖 Architecture Router"}
    
    F -->|Raw Scalars| G["🏷️ Simple RNN (No Embedding)"]
    F -->|16D Vectors| H["🗺️ Simple RNN (With Embedding)"]
    F -->|Gated Gates| I["🧠 LSTM Model (With Embedding)"]
    F -->|Gated Gates| J["⚡ GRU Model (With Embedding)"]
    
    G & H & I & J --> K["🎯 Dense Output Layer (Sigmoid)"]
    K -->|Probs [0.0 - 1.0]| L["📊 Live Web Dashboard UI"]
```

---

## 👥 Model Personas

Here is a fun, intuitive breakdown of each model's "personality" and technical configuration:

### 🏷️ 1. Simple RNN (Without Embedding)
> *"The Index Mapper"*
* **The Gimmick:** Attempts to classify sentiment by treating raw vocabulary numbers (e.g. "love" = `98`, "boring" = `88`) as direct scalar activations.
* **Accuracy:** **~51%** (Equal to flipping a coin).
* **The Lesson:** Since index numbers are arbitrary, the model sees no correlation between the value of a token index and its emotional polarity.

### 🗺️ 2. Simple RNN (With Embedding)
> *"The Vector Navigator"*
* **The Gimmick:** Maps word indices into a learnable **16-dimensional continuous vector space**. Similar words gather in clusters.
* **Accuracy:** **~99%** (Learns the training sentences easily).
* **The Lesson:** Words gain context! However, standard recurrent cells suffer from fading state values over longer sentences.

### 🧠 3. LSTM (With Embedding)
> *"The Gatekeeper"*
* **The Gimmick:** Incorporates **Long Short-Term Memory** structures using input, forget, and output gates to manage cell state memory.
* **Accuracy:** **100%** (Perfect retention).
* **The Lesson:** By regulating what to remember and what to discard, LSTMs retain sentiment context across sequential tokens flawlessly.

### ⚡ 4. GRU (With Embedding)
> *"The Speedster"*
* **The Gimmick:** A streamlined version of the LSTM that merges the cell state and hidden state, using only reset and update gates.
* **Accuracy:** **100%** (Fast and precise).
* **The Lesson:** Achieves the same high accuracy as LSTMs but with fewer parameters and shorter training times.

---

## 📈 Model Specifications Table

| Metric / Feature | Simple RNN (Raw) | Simple RNN (Emb) | LSTM Model | GRU Model |
| :--- | :--- | :--- | :--- | :--- |
| **Embedding Layer** | None (1D Scalar) | Learnable (16D) | Learnable (16D) | Learnable (16D) |
| **Recurrent Units** | 16 Units | 16 Units | 16 Units | 16 Units |
| **Trainable Params**| ~300 | ~4,000 | ~6,000 | ~5,200 |
| **Training Accuracy**| **51.00%** | **99.00%** | **100.00%** | **100.00%** |
| **Recurrent Cell** | `SimpleRNN` | `SimpleRNN` | `LSTM` | `GRU` |

---

## 📸 Dashboard Screenshots

### 1. Interactive Sentiment Tester & Tokenizer Pipeline
![Dashboard Interface](images/dashboard_main.png)

### 2. Architecture Specifications & Metrics
![Specifications Table](images/dashboard_specs.png)

---

## 💡 Key Deep Learning Insights

> [!TIP]
> ### 💡 Why pre-padding ('pre') is critical for RNNs
> If we pad sequences at the end (**post-padding**), the final recurrent steps process long sequences of `0` padding inputs. For Simple RNNs, this causes the gradient of the actual words to fade out by the time the model reaches the final prediction step. **Pre-padding** places the zeros first, ensuring the final activations are driven by the actual words.

> [!NOTE]
> ### 🔍 Handling Out-Of-Vocabulary (OOV) words
> Our preprocessor tokenizes raw input and replaces unknown words with the `<OOV>` token (mapped to index `1`). This ensures the model does not crash when encountering unseen inputs like *"keyboard"* or *"elephant"*, mapping them to a neutral background weight.

---

## 🚀 Interactive Prompt Sandbox
Launch the local web server and try entering these sentences to compare predictions:

* **Standard Positive:** *"I had a wonderful time, highly recommended!"*
  * *Result:* LSTM & GRU predict POS (~1.00). Simple RNN (Raw) remains uncertain (~0.50).
* **Mixed Sentiments:** *"This movie was boring but the acting was fantastic."*
  * *Result:* Watch LSTMs and GRUs evaluate the sequence weight, while the Simple RNN with embedding struggles on mixed feedback.
* **Gibberish / OOV:** *"An elephant keyboard is running."*
  * *Result:* The tokens map to index `1` (`<OOV>`). Models output a neutral background probability.

---

## 🛠️ Installation & Run Commands

### 1. Install Dependencies
```bash
pip install tensorflow numpy flask requests
```

### 2. Train and Save Assets
Generates model binaries and token configurations:
```bash
python3 train_and_save.py
```

### 3. Launch Flask Server
```bash
python3 app.py
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:5001](http://127.0.0.1:5001)**
