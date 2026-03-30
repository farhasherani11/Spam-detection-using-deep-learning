import os
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models")

# LOAD DATA
print("📂 Loading features...")

X_train, X_test, y_train, y_test = pickle.load(
    open(os.path.join(model_path, "features.pkl"), "rb")
)

# LOAD MODEL
print("🤖 Loading LSTM model...")

model = load_model(os.path.join(model_path, "lstm_model.h5"), compile=False)

# PREDICT
print("🔍 Evaluating...")

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# RESULTS
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))

print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\n📌 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))