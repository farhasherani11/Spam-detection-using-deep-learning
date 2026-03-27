
# 1. IMPORT LIBRARIES

import pandas as pd
import re
import nltk
import os
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 2. LOAD DATA
df = pd.read_csv("D:/Spam-detection-using-deep-learning/data/raw/spam.csv", encoding='latin-1')

# Rename columns
df = df.rename(columns={
    'Category': 'v1',
    'Message': 'v2'
})

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("Data Loaded:\n", df.head())

# 3. SPAM TYPE CLASSIFICATION

def classify_spam_type(text, label):
    text = text.lower()
    
    if label == "ham":
        return "ham"
    
    if "win" in text or "prize" in text or "lottery" in text:
        return "lottery"
    elif "loan" in text or "bank" in text or "money" in text:
        return "financial"
    elif "click" in text or "link" in text or "verify" in text:
        return "phishing"
    elif "job" in text or "earn" in text:
        return "job_spam"
    elif "otp" in text or "password" in text:
        return "otp_fraud"
    elif "free" in text or "offer" in text or "buy" in text:
        return "promotion"
    elif "sex" in text or "adult" in text:
        return "adult"
    else:
        return "general_spam"


df['spam_type'] = df.apply(lambda x: classify_spam_type(x['message'], x['label']), axis=1)

print("\nSpam Type Added:\n", df[['label', 'spam_type']].head())


# 4. TEXT PREPROCESSING
# Download only if not present
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)


df['cleaned_message'] = df['message'].apply(clean_text)

print("\nCleaned Text:\n", df[['message', 'cleaned_message']].head())

# 5. ENCODE LABELS

le = LabelEncoder()
df['spam_type_encoded'] = le.fit_transform(df['spam_type'])

print("\nEncoded Classes:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# 6. TOKENIZATION + PADDING
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['cleaned_message'])

X = tokenizer.texts_to_sequences(df['cleaned_message'])
X = pad_sequences(X, maxlen=100)


# Binary
y_binary = df['label'].map({'ham': 0, 'spam': 1})

# Multi-class (recommended)
y_multi = df['spam_type_encoded']



# 8. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi, test_size=0.2, random_state=42
)

print("\nTrain Test Split Done")
print("X_train shape:", X_train.shape)


# 9. SAVE FILES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models")
data_path = os.path.join(BASE_DIR, "..", "data")

os.makedirs(model_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

# Save tokenizer
pickle.dump(tokenizer, open(os.path.join(model_path, "tokenizer.pkl"), "wb"))

# Save label encoder
pickle.dump(le, open(os.path.join(model_path, "label_encoder.pkl"), "wb"))

# Save processed dataset
df.to_csv(os.path.join(data_path, "processed_data.csv"), index=False)

print("\n Preprocessing completed & data saved successfully!")