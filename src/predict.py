import os
import pickle
import numpy as np
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK (first time only)
nltk.download('stopwords')
nltk.download('wordnet')

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models")

# LOAD MODEL + TOKENIZER
model = load_model(os.path.join(model_path, "lstm_model.h5"))
tokenizer = pickle.load(open(os.path.join(model_path, "tokenizer.pkl"), "rb"))
label_encoder = pickle.load(open(os.path.join(model_path, "label_encoder.pkl"), "rb"))

# TEXT CLEANING
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# PREDICT FUNCTION
def predict_message(message):
    cleaned = clean_text(message)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)

    pred = model.predict(padded)
    label = np.argmax(pred)

    result = label_encoder.inverse_transform([label])[0]
    confidence = np.max(pred)

    return result, confidence

# USER INPUT LOOP
print("📩 Spam Detection CLI")
print("Type 'exit' to quit\n")

while True:
    msg = input("Enter message: ")

    if msg.lower() == "exit":
        break

    result, confidence = predict_message(msg)

    if result == "ham":
        print(f"✅ Not Spam (Ham) | Confidence: {confidence:.2f}\n")
    else:
        print(f"🚨 Spam Detected: {result} | Confidence: {confidence:.2f}\n")