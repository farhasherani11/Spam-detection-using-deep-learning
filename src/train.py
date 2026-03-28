import os
import pickle
import numpy as np
import pandas as pd

from model_lstm import build_lstm_model
from bert import load_bert_model

# PATH SETUP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "..", "data")
model_path = os.path.join(BASE_DIR, "..", "models")

os.makedirs(model_path, exist_ok=True)

# LOAD LSTM FEATURES

print(" Loading LSTM features...")

features_file = os.path.join(model_path, "features.pkl")

X_train, X_test, y_train, y_test = pickle.load(open(features_file, "rb"))

print(" Features Loaded!")
print("X_train shape:", X_train.shape)

num_classes = len(set(y_train))


# TRAIN LSTM MODEL
print("\n Training LSTM...")

lstm_model = build_lstm_model(
    input_length=X_train.shape[1],
    vocab_size=5000,
    num_classes=num_classes
)

lstm_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save LSTM model
lstm_model.save(os.path.join(model_path, "lstm_model.h5"))

print(" LSTM Model Saved!")


# LOAD ORIGINAL DATA FOR BERT

print("\n Loading data for BERT...")

csv_path = os.path.join(data_path, "processed_data.csv")
df = pd.read_csv(csv_path)

df['cleaned_message'] = df['cleaned_message'].fillna("").astype(str)

# Load label encoder
le = pickle.load(open(os.path.join(model_path, "label_encoder.pkl"), "rb"))

y = le.transform(df['spam_type'])


# TRAIN BERT MODEL
print("\n Training BERT...")

bert_model, tokenizer = load_bert_model(num_classes)

# Tokenize text
bert_inputs = tokenizer(
    list(df['cleaned_message']),
    padding=True,
    truncation=True,
    max_length=100,
    return_tensors='tf'
)

bert_model.fit(
    dict(bert_inputs),
    y,
    epochs=1,
    batch_size=8
)

# Save BERT model + tokenizer
bert_model.save_pretrained(os.path.join(model_path, "bert_model"))
tokenizer.save_pretrained(os.path.join(model_path, "bert_model"))

print(" BERT Model Saved!")

print("\n TRAINING COMPLETE!")