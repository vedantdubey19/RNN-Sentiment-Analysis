import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Paths
MODELS_DIR = "models"
NO_EMB_PATH = os.path.join(MODELS_DIR, "model_no_embedding.keras")
WITH_EMB_PATH = os.path.join(MODELS_DIR, "model_with_embedding.keras")
LSTM_PATH = os.path.join(MODELS_DIR, "model_lstm.keras")
GRU_PATH = os.path.join(MODELS_DIR, "model_gru.keras")
VOCAB_PATH = os.path.join(MODELS_DIR, "tokenizer_vocab.json")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

# Global states
models = {}
tokenizer_data = {}
metadata = {}

def load_resources():
    global models, tokenizer_data, metadata
    
    # Load metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
        
    # Load tokenizer vocabulary
    with open(VOCAB_PATH, "r") as f:
        tokenizer_data = json.load(f)
        
    # Load models
    models["no_embedding"] = load_model(NO_EMB_PATH)
    models["with_embedding"] = load_model(WITH_EMB_PATH)
    models["lstm"] = load_model(LSTM_PATH)
    models["gru"] = load_model(GRU_PATH)
    print("All models and tokenizers loaded successfully!")

# Call resource loader
load_resources()

def tokenize_and_pad(text):
    """
    Manual implementation of Keras Tokenizer texts_to_sequences and pad_sequences 
    to ensure perfect synchronization and no dependence on external keras files in flask.
    """
    word_index = tokenizer_data.get("word_index", {})
    oov_token = tokenizer_data.get("oov_token", "<OOV>")
    max_len = metadata.get("max_len", 7)
    
    # Standardize text: lower-case and split by spaces
    words = text.lower().strip().split()
    
    # Map words to indices
    seq = []
    tokens_mapping = []
    
    for word in words:
        # Strip punctuation
        clean_word = "".join(c for c in word if c.isalnum() or c == "'")
        if not clean_word:
            continue
        idx = word_index.get(clean_word, word_index.get(oov_token, 1))
        seq.append(idx)
        tokens_mapping.append({"token": word, "index": idx})
        
    # Pad sequence ('pre' padding)
    if len(seq) < max_len:
        padding_needed = max_len - len(seq)
        padded_seq = [0] * padding_needed + seq
        visual_sequence = [{"token": "<PAD>", "index": 0}] * padding_needed + tokens_mapping
    else:
        padded_seq = seq[-max_len:]
        visual_sequence = tokens_mapping[-max_len:]
        
    return padded_seq, visual_sequence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
            
        text = data["text"]
        padded_seq, tokens_mapping = tokenize_and_pad(text)
        
        # Format input arrays
        input_emb = np.array([padded_seq])  # (1, max_len)
        input_no_emb = input_emb.reshape((1, len(padded_seq), 1))  # (1, max_len, 1)
        
        # Predict on all models
        pred_no_emb = float(models["no_embedding"].predict(input_no_emb, verbose=0)[0][0])
        pred_with_emb = float(models["with_embedding"].predict(input_emb, verbose=0)[0][0])
        pred_lstm = float(models["lstm"].predict(input_emb, verbose=0)[0][0])
        pred_gru = float(models["gru"].predict(input_emb, verbose=0)[0][0])
        
        response = {
            "text": text,
            "tokens_mapping": tokens_mapping,
            "predictions": {
                "no_embedding": {
                    "probability": pred_no_emb,
                    "sentiment": "Positive" if pred_no_emb >= 0.5 else "Negative",
                    "confidence": abs(pred_no_emb - 0.5) * 2  # scale 0 to 1
                },
                "with_embedding": {
                    "probability": pred_with_emb,
                    "sentiment": "Positive" if pred_with_emb >= 0.5 else "Negative",
                    "confidence": abs(pred_with_emb - 0.5) * 2
                },
                "lstm": {
                    "probability": pred_lstm,
                    "sentiment": "Positive" if pred_lstm >= 0.5 else "Negative",
                    "confidence": abs(pred_lstm - 0.5) * 2
                },
                "gru": {
                    "probability": pred_gru,
                    "sentiment": "Positive" if pred_gru >= 0.5 else "Negative",
                    "confidence": abs(pred_gru - 0.5) * 2
                }
            }
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
