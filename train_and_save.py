import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define a rich sentiment dataset for robust training
sentences = [
    # Positive reviews
    "I love this course", "This is amazing", "Absolutely fantastic movie", "I had a wonderful time",
    "Highly recommended for everyone", "This is brilliant work", "Superb acting and great story",
    "The plot was very engaging", "I thoroughly enjoyed this", "Outstanding performance by the lead",
    "An excellent piece of cinema", "A true masterpiece of our time", "Very positive experience indeed",
    "Loved the cinematography and direction", "So inspiring and beautiful", "A delightful watch",
    "Perfect in every possible way", "Great value and highly useful", "This exceeded all my expectations",
    "It makes me so happy", "Very friendly staff and good service", "Clean and neat environment",
    "Definitely worth buying", "The best purchase I have ever made", "An incredibly fun experience",
    "Highly satisfied with the product", "Quick and helpful support", "Stunning visuals and audio",
    "A heartwarming and sweet story", "Top tier quality and design", "Highly educational and clear",
    "I would buy this again without hesitation", "Beautifully crafted and designed", "Surprisingly good and solid",
    "A breath of fresh air", "Remarkable attention to detail", "Simply wonderful", "An absolute joy to read",
    "A triumph of filmmaking", "One of my absolute favorites", "It works perfectly and is fast",
    "Very easy to use and intuitive", "A very pleasant surprise", "Highly recommended book",
    "Super fast shipping and great package", "Exactly what I wanted", "A masterpiece of storytelling",
    "Incredible performance and value", "Highly impressed with this", "I will definitely recommend it",

    # Negative reviews
    "I hate this topic", "This is boring", "Absolutely terrible movie", "I had a horrible time",
    "Not recommended at all", "This is awful work", "Bad acting and poor story",
    "The plot was very dull", "I did not enjoy this at all", "Disappointing performance by the lead",
    "A terrible piece of cinema", "A complete waste of time", "Very negative experience indeed",
    "Hated the direction and pacing", "So boring and slow", "A painful watch",
    "Worst movie I have ever seen", "Completely useless and expensive", "This fell short of expectations",
    "It makes me so angry and sad", "Unprofessional staff and slow service", "Dirty and messy environment",
    "Definitely not worth the money", "The worst purchase I have ever made", "An incredibly frustrating experience",
    "Highly dissatisfied with the quality", "Rude and unhelpful customer support", "Awful visuals and terrible audio",
    "A tragic and depressing failure", "Low quality and cheap material", "Highly confusing and unclear instructions",
    "I would never buy this again", "Poorly designed and cheap looking", "Surprisingly bad and broken",
    "A waste of resources", "Total lack of attention to detail", "Simply awful", "An absolute chore to read",
    "A failure of filmmaking", "One of my least favorites", "It crashes constantly and is slow",
    "Very hard to use and frustrating", "A very unpleasant surprise", "Avoid this book at all costs",
    "Extremely slow shipping and broken package", "Not what I expected", "A failure of storytelling",
    "Poor performance and overpriced", "Extremely disappointed with this", "I will never recommend it"
]

labels = np.array([1]*50 + [0]*50)  # 50 positive (1), 50 negative (0)

# Tokenizer configuration with Out-Of-Vocabulary (OOV) token
oov_tok = "<OOV>"
tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)

# Save tokenizer configuration to JSON for the Flask app
vocab_size = len(tokenizer.word_index) + 1
tokenizer_json = {
    "word_index": tokenizer.word_index,
    "oov_token": oov_tok
}

os.makedirs("models", exist_ok=True)
with open("models/tokenizer_vocab.json", "w") as f:
    json.dump(tokenizer_json, f, indent=4)

# Convert texts to sequences and pad ('pre' padding is critical for SimpleRNN performance)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='pre')
max_len = padded.shape[1]

# Save metadata for frontend/backend alignment
metadata = {
    "max_len": max_len,
    "vocab_size": vocab_size
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Dataset Size: {len(sentences)} samples")
print(f"Vocabulary Size: {vocab_size}")
print(f"Max Sequence Length: {max_len}")

# --- MODEL 1: Simple RNN WITHOUT Embedding ---
model_no_embedding = Sequential([
    Input(shape=(max_len, 1)),
    SimpleRNN(16),
    Dense(1, activation='sigmoid')
])
model_no_embedding.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
padded_reshaped = padded.reshape((padded.shape[0], padded.shape[1], 1))

print("\nTraining Model 1: Simple RNN WITHOUT Embedding...")
model_no_embedding.fit(padded_reshaped, labels, epochs=80, verbose=0)
loss_ne, acc_ne = model_no_embedding.evaluate(padded_reshaped, labels, verbose=0)
print(f"Model 1 (No Embedding) Training Accuracy: {acc_ne * 100:.2f}%")

# --- MODEL 2: Simple RNN WITH Embedding ---
model_with_embedding = Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=vocab_size, output_dim=16),
    SimpleRNN(16),
    Dense(1, activation='sigmoid')
])
model_with_embedding.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining Model 2: Simple RNN WITH Embedding...")
model_with_embedding.fit(padded, labels, epochs=80, verbose=0)
loss_we, acc_we = model_with_embedding.evaluate(padded, labels, verbose=0)
print(f"Model 2 (With Embedding) Training Accuracy: {acc_we * 100:.2f}%")

# --- MODEL 3: LSTM WITH Embedding ---
model_lstm = Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=vocab_size, output_dim=16),
    LSTM(16),
    Dense(1, activation='sigmoid')
])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining Model 3: LSTM WITH Embedding...")
model_lstm.fit(padded, labels, epochs=80, verbose=0)
loss_lstm, acc_lstm = model_lstm.evaluate(padded, labels, verbose=0)
print(f"Model 3 (LSTM) Training Accuracy: {acc_lstm * 100:.2f}%")

# --- MODEL 4: GRU WITH Embedding ---
model_gru = Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=vocab_size, output_dim=16),
    GRU(16),
    Dense(1, activation='sigmoid')
])
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining Model 4: GRU WITH Embedding...")
model_gru.fit(padded, labels, epochs=80, verbose=0)
loss_gru, acc_gru = model_gru.evaluate(padded, labels, verbose=0)
print(f"Model 4 (GRU) Training Accuracy: {acc_gru * 100:.2f}%")

# Save all models
model_no_embedding.save("models/model_no_embedding.h5")
model_with_embedding.save("models/model_with_embedding.h5")
model_lstm.save("models/model_lstm.h5")
model_gru.save("models/model_gru.h5")
print("\nAll models and configuration files saved successfully in 'models/' directory!")
