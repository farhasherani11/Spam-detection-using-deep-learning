import pandas as pd

df = pd.read_csv("D:/Spam-detection-using-deep-learning/data/raw/spam.csv", encoding='latin-1')
import pandas as pd

df = pd.read_csv("D:/Spam-detection-using-deep-learning/data/raw/spam.csv", encoding='latin-1')

# Rename columns
df = df.rename(columns={
    'Category': 'v1',
    'Message': 'v2'
})


df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())

spam_categories = [
    "promotion",     # ads, offers, discounts
    "financial",     # bank, loan, money
    "phishing",      # fake links, login traps
    "lottery",       # win prize, lucky draw
    "job_spam",      # job offers, work from home
    "otp_fraud",     # OTP/password scams
    "adult",         # inappropriate content
    "general_spam",  # spam but no clear type
    "ham"            # normal message
]
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


#print(df.head(15))

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
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

print(df[['message', 'cleaned_message']].head())



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['spam_type_encoded'] = le.fit_transform(df['spam_type'])

print(dict(zip(le.classes_, le.transform(le.classes_))))



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['cleaned_message'])

X = tokenizer.texts_to_sequences(df['cleaned_message'])
X = pad_sequences(X, maxlen=100)



# Binary (spam / ham)
y_binary = df['label'].map({'ham': 0, 'spam': 1})

# Multi-class (spam categories)
y_multi = df['spam_type_encoded']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)




import pickle
import os

os.makedirs("../models", exist_ok=True)

# Save tokenizer
pickle.dump(tokenizer, open("../models/tokenizer.pkl", "wb"))

# Save processed data
df.to_csv("../data/processed_data.csv", index=False)

print("Preprocessing completed & data saved ")