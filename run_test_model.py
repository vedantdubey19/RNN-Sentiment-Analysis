import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model assets
MODELS_DIR = "models"
NO_EMB_PATH = os.path.join(MODELS_DIR, "model_no_embedding.keras")
WITH_EMB_PATH = os.path.join(MODELS_DIR, "model_with_embedding.keras")
LSTM_PATH = os.path.join(MODELS_DIR, "model_lstm.keras")
GRU_PATH = os.path.join(MODELS_DIR, "model_gru.keras")
VOCAB_PATH = os.path.join(MODELS_DIR, "tokenizer_vocab.json")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

print("Loading saved models and configurations...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
with open(VOCAB_PATH, "r") as f:
    tokenizer_data = json.load(f)

model_no_emb = load_model(NO_EMB_PATH)
model_with_emb = load_model(WITH_EMB_PATH)
model_lstm = load_model(LSTM_PATH)
model_gru = load_model(GRU_PATH)
print("All models loaded successfully.\n")

def preprocess_text(text):
    word_index = tokenizer_data["word_index"]
    oov_token = tokenizer_data["oov_token"]
    max_len = metadata["max_len"]
    
    words = text.lower().strip().split()
    seq = []
    for word in words:
        clean_word = "".join(c for c in word if c.isalnum() or c == "'")
        if not clean_word:
            continue
        idx = word_index.get(clean_word, word_index.get(oov_token, 1))
        seq.append(idx)
        
    if len(seq) < max_len:
        padded_seq = [0] * (max_len - len(seq)) + seq
    else:
        padded_seq = seq[-max_len:]
        
    return padded_seq

# Test cases
test_sentences = [
    "I love this course",
    "This is boring",
    "Absolutely terrible movie",
    "A wonderful brilliant masterpiece",
    "Some completely unknown words like keyboard elephant"
]

print("=" * 100)
print(f"{'TEST SENTENCE':<45} | {'NO EMBED':<9} | {'WITH EMBED':<10} | {'LSTM':<9} | {'GRU':<9}")
print("-" * 100)

for text in test_sentences:
    padded_seq = preprocess_text(text)
    
    # Format inputs
    input_emb = np.array([padded_seq])
    input_no_emb = input_emb.reshape((1, len(padded_seq), 1))
    
    # Predictions
    p_no_emb = float(model_no_emb.predict(input_no_emb, verbose=0)[0][0])
    p_with_emb = float(model_with_emb.predict(input_emb, verbose=0)[0][0])
    p_lstm = float(model_lstm.predict(input_emb, verbose=0)[0][0])
    p_gru = float(model_gru.predict(input_emb, verbose=0)[0][0])
    
    # Format output columns
    no_emb_str = f"{'POS' if p_no_emb >= 0.5 else 'NEG'} ({p_no_emb:.2f})"
    with_emb_str = f"{'POS' if p_with_emb >= 0.5 else 'NEG'} ({p_with_emb:.2f})"
    lstm_str = f"{'POS' if p_lstm >= 0.5 else 'NEG'} ({p_lstm:.2f})"
    gru_str = f"{'POS' if p_gru >= 0.5 else 'NEG'} ({p_gru:.2f})"
    
    # Trim sentence print length
    disp_text = text if len(text) <= 45 else text[:42] + "..."
    print(f"{disp_text:<45} | {no_emb_str:<9} | {with_emb_str:<10} | {lstm_str:<9} | {gru_str:<9}")

print("=" * 100)
