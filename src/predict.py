import os
import pickle
import numpy as np
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# =========================
# 📥 DOWNLOAD NLTK
# =========================
nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# 📂 PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models")

# =========================
# 🤖 LOAD MODEL
# =========================
print("🤖 Loading model...")

model = load_model(
    os.path.join(model_path, "lstm_model.h5"),
    compile=False
)

tokenizer = pickle.load(open(os.path.join(model_path, "tokenizer.pkl"), "rb"))
label_encoder = pickle.load(open(os.path.join(model_path, "label_encoder.pkl"), "rb"))

print("✅ Model Loaded!\n")

# =========================
# 🧹 TEXT CLEANING
# =========================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()

    # 🔥 KEEP numbers + money symbols
    text = re.sub(r'[^a-zA-Z0-9₹$]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# =========================
# 🔍 PREDICT FUNCTION
# =========================
def predict_message(message):
    cleaned = clean_text(message)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)

    pred = model.predict(padded, verbose=0)[0]

    label_index = int(np.argmax(pred))
    result = label_encoder.inverse_transform([label_index])[0]
    confidence = float(np.max(pred))

    return result, confidence

# =========================
# 🚨 SPAM KEYWORDS (BOOST)
# =========================
spam_keywords = [
    "win", "lottery", "prize", "offer", "free",
    "loan", "click", "verify", "otp", "earn",
    "money", "urgent", "account", "bank",
    "discount", "buy", "winner", "cash"
]

# =========================
# 🖥️ CLI LOOP
# =========================
print("📩 Spam Detection CLI")
print("Type 'exit' to quit\n")

while True:
    msg = input("Enter message: ").strip()

    if msg.lower() == "exit":
        print("👋 Exiting...")
        break

    if msg == "":
        print("⚠️ Please enter a valid message\n")
        continue

    result, confidence = predict_message(msg)

    # 🔥 KEYWORD CHECK
    keyword_flag = any(word in msg.lower() for word in spam_keywords)

    # =========================
    # 🎯 FINAL DECISION LOGIC
    # =========================
    if result == "ham":
        if confidence < 0.6 or keyword_flag:
            print("⚠️ Suspicious Message (Possible Spam)")
        else:
            print("✅ Not Spam (Ham)")

    else:
        print(f"🚨 Spam Detected: {result}")

    print(f"📊 Confidence: {confidence:.2f}\n")