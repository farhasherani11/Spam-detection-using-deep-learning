from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_lstm_model(input_length, vocab_size, num_classes):

    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=128))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',  # ✅ FIX
        optimizer='adam',
        metrics=['accuracy']
    )

    return model