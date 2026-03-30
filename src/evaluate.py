import os
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from transformers import TFBertForSequenceClassification, BertTokenizer

# =========================
# 📂 PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models")
data_path = os.path.join(BASE_DIR, "..", "data")

# =========================
# 📥 LOAD LABEL ENCODER
# =========================
label_encoder = pickle.load(
    open(os.path.join(model_path, "label_encoder.pkl"), "rb")
)

# ============================================================
# 🔷 LSTM EVALUATION
# ============================================================
print("\n=========================")
print("🔷 LSTM EVALUATION")
print("=========================\n")

print("📂 Loading features...")

X_train, X_test, y_train, y_test = pickle.load(
    open(os.path.join(model_path, "features.pkl"), "rb")
)

print("🤖 Loading LSTM model...")

lstm_model = load_model(
    os.path.join(model_path, "lstm_model.h5"),
    compile=False
)

print("🔍 Evaluating LSTM...")

y_pred_lstm = lstm_model.predict(X_test, verbose=0)
y_pred_lstm = np.argmax(y_pred_lstm, axis=1)

# Fix label mismatch
unique_labels = np.unique(y_test)
target_names = [label_encoder.classes_[i] for i in unique_labels]

print("\n📊 LSTM Accuracy:", accuracy_score(y_test, y_pred_lstm))

print("\n📄 LSTM Classification Report:\n")
print(
    classification_report(
        y_test,
        y_pred_lstm,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0
    )
)

print("\n📌 LSTM Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_lstm))


# ============================================================
# 🤖 BERT EVALUATION (MEMORY SAFE - BATCHED)
# ============================================================
print("\n\n=========================")
print("🤖 BERT EVALUATION")
print("=========================\n")

print("📂 Loading data for BERT...")

df = pd.read_csv(os.path.join(data_path, "processed_data.csv"))
df['cleaned_message'] = df['cleaned_message'].fillna("").astype(str)

y = label_encoder.transform(df['spam_type'])

print("🤖 Loading BERT model...")

bert_model = TFBertForSequenceClassification.from_pretrained(
    os.path.join(model_path, "bert_model")
)

tokenizer = BertTokenizer.from_pretrained(
    os.path.join(model_path, "bert_model")
)

print("🔍 Evaluating BERT (batch mode)...")

batch_size = 32   # 🔥 change to 16 if RAM issue

y_pred_bert = []

for i in range(0, len(df), batch_size):
    batch_texts = df['cleaned_message'][i:i+batch_size].tolist()

    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=100,
        return_tensors="tf"
    )

    outputs = bert_model(inputs)
    preds = np.argmax(outputs.logits.numpy(), axis=1)

    y_pred_bert.extend(preds)

y_pred_bert = np.array(y_pred_bert)

# Fix label mismatch
unique_labels_bert = np.unique(y)
target_names_bert = [label_encoder.classes_[i] for i in unique_labels_bert]

print("\n📊 BERT Accuracy:", accuracy_score(y, y_pred_bert))

print("\n📄 BERT Classification Report:\n")
print(
    classification_report(
        y,
        y_pred_bert,
        labels=unique_labels_bert,
        target_names=target_names_bert,
        zero_division=0
    )
)

print("\n📌 BERT Confusion Matrix:\n")
print(confusion_matrix(y, y_pred_bert))


# ============================================================
# 🎯 DONE
# ============================================================
print("\n🎉 EVALUATION COMPLETE!")