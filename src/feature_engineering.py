
import pandas as pd
import os
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# LOAD DATA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "..", "data", "processed_data.csv")
model_path = os.path.join(BASE_DIR, "..", "models")

df = pd.read_csv(data_path)

print(" Processed Data Loaded:", df.shape)

# LABEL ENCODING
le = LabelEncoder()
df['spam_type_encoded'] = le.fit_transform(df['spam_type'])

print(" Labels Encoded")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# TOKENIZATION
df['cleaned_message'] = df['cleaned_message'].fillna("")
df['cleaned_message'] = df['cleaned_message'].astype(str)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['cleaned_message'])

X = tokenizer.texts_to_sequences(df['cleaned_message'])

# PADDING
X = pad_sequences(X, maxlen=100)

print(" Text Converted to Sequences")
y = df['spam_type_encoded']

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(" Train-Test Split Done")
print("X_train shape:", X_train.shape)

os.makedirs(model_path, exist_ok=True)

# Save tokenizer
pickle.dump(tokenizer, open(os.path.join(model_path, "tokenizer.pkl"), "wb"))

# Save label encoder
pickle.dump(le, open(os.path.join(model_path, "label_encoder.pkl"), "wb"))

# Save features
pickle.dump((X_train, X_test, y_train, y_test),
            open(os.path.join(model_path, "features.pkl"), "wb"))

print("\nFeature Engineering Completed & Saved!")