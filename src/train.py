import os
import pickle
import numpy as np
import pandas as pd

from model_lstm import build_lstm_model
from bert import load_bert_model

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# =========================
# 📁 PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "..", "data")
model_path = os.path.join(BASE_DIR, "..", "models")

os.makedirs(model_path, exist_ok=True)

# =========================
# 📂 LOAD LSTM FEATURES
# =========================
print("📂 Loading LSTM features...")

features_file = os.path.join(model_path, "features.pkl")

X_train, X_test, y_train, y_test = pickle.load(open(features_file, "rb"))

print("✅ Features Loaded!")
print("X_train shape:", X_train.shape)

num_classes = len(set(y_train))

# =========================
# ⚖️ CLASS WEIGHTS (IMPORTANT)
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("⚖️ Class Weights:", class_weights)

# =========================
# 🧠 TRAIN LSTM MODEL
# =========================
print("\n🚀 Training LSTM...")

lstm_model = build_lstm_model(
    input_length=X_train.shape[1],
    vocab_size=8000,  # 🔥 increased vocab
    num_classes=num_classes
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

lstm_model.fit(
    X_train,
    y_train,
    epochs=10,  # 🔥 increased
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# Save LSTM model
lstm_model.save(os.path.join(model_path, "lstm_model.h5"))

print("✅ LSTM Model Saved!")

# =========================
# 📂 LOAD DATA FOR BERT
# =========================
print("\n📂 Loading data for BERT...")

csv_path = os.path.join(data_path, "processed_data.csv")
df = pd.read_csv(csv_path)

df['cleaned_message'] = df['cleaned_message'].fillna("").astype(str)

# Load label encoder
le = pickle.load(open(os.path.join(model_path, "label_encoder.pkl"), "rb"))

y = le.transform(df['spam_type'])

# =========================
# 🤖 TRAIN BERT MODEL
# =========================
print("\n🚀 Training BERT...")

bert_model, tokenizer = load_bert_model(num_classes)

# Tokenize
bert_inputs = tokenizer(
    list(df['cleaned_message']),
    padding=True,
    truncation=True,
    max_length=100,
    return_tensors='tf'
)

# Train
bert_model.fit(
    dict(bert_inputs),
    y,
    epochs=2,  # 🔥 increased (safe)
    batch_size=8,
    validation_split=0.1,
    verbose=1
)

# Save BERT model
bert_model.save_pretrained(os.path.join(model_path, "bert_model"))
tokenizer.save_pretrained(os.path.join(model_path, "bert_model"))

print("✅ BERT Model Saved!")

# =========================
# 🎉 DONE
# =========================
print("\n🎯 TRAINING COMPLETE!")