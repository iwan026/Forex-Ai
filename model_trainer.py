import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import config

def build_model(input_shape):
    """Bangun model LSTM sederhana"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train():
    # Load data yang sudah diproses
    with open('data/processed_data.pkl', 'rb') as f:
        X, y = pickle.load(f)
    
    # Split data train/test
    split = int(len(X) * (1 - config.TEST_SIZE))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Bangun dan train model
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_test, y_test)
    )
    
    # Simpan model
    model.save('model/lstm_model.h5')

if __name__ == "__main__":
    train()