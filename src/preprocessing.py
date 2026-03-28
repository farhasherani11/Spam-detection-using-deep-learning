

#  IMPORT LIBRARIES
import pandas as pd
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# 2. LOAD DATA
df = pd.read_csv("D:/Spam-detection-using-deep-learning/data/raw/spam.csv", encoding='latin-1')

# Rename columns
df = df.rename(columns={
    'Category': 'label',
    'Message': 'message'
})

df = df[['label', 'message']]

print(" Data Loaded")
print(df.head())


# 3. SPAM TYPE CLASSIFICATION
def classify_spam_type(text, label):
    text = str(text).lower()

    if label == "ham":
        return "ham"
    elif "win" in text or "prize" in text or "lottery" in text:
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


df['spam_type'] = df.apply(
    lambda x: classify_spam_type(x['message'], x['label']),
    axis=1
)

print("\n Spam Type Added")
print(df[['label', 'spam_type']].head())


#  TEXT CLEANING

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
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


df['cleaned_message'] = df['message'].apply(clean_text)

print("\n Text Cleaned")
print(df[['message', 'cleaned_message']].head())


#SAVE CLEAN DATA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data")

os.makedirs(data_path, exist_ok=True)

df.to_csv(os.path.join(data_path, "processed_data.csv"), index=False)

print("\n Preprocessing Completed & Saved!")