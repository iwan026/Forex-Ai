import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from data_processor import load_data, preprocess_data

def predict_signal():
    # Load model dan scaler
    model = tf.keras.models.load_model('model/lstm_model.h5')
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Ambil data terbaru (contoh: 60 candle terakhir)
    df = load_data('data/EURUSD_H1.csv').tail(60)
    processed_data = scaler.transform(df[['open', 'high', 'low', 'close']])
    
    # Reshape untuk prediksi (1 sample, 60 timesteps, 4 features)
    input_data = processed_data[np.newaxis, ...]
    
    # Prediksi
    prediction = model.predict(input_data)[0][0]
    signal = "BUY" if prediction > 0.5 else "SELL"
    
    print(f"Signal: {signal} | Confidence: {prediction*100:.2f}%")

if __name__ == "__main__":
    predict_signal()