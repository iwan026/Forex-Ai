import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import config
import pickle

def load_data(filepath):
    """Load data dari CSV"""
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    return df

def create_target(df):
    """Buat target: 1 jika harga naik di candle berikutnya"""
    df['next_close'] = df['close'].shift(-1)
    df[config.TARGET] = (df['next_close'] > df['close']).astype(int)
    return df.dropna()

def preprocess_data(df):
    """Normalisasi data dan siapkan untuk LSTM"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[config.FEATURES])
    
    # Simpan scaler untuk prediksi nanti
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return scaled_data

def create_sequences(data):
    """Buat sequences untuk LSTM"""
    X, y = [], []
    for i in range(config.LOOKBACK_WINDOW, len(data)):
        X.append(data[i-config.LOOKBACK_WINDOW:i])
        y.append(data[i-1, -1])  # Ambil target dari kolom terakhir
    return np.array(X), np.array(y)

if __name__ == "__main__":
    df = load_data('data/EURUSD_H1.csv')
    df = create_target(df)
    processed_data = preprocess_data(df)
    X, y = create_sequences(processed_data)
    
    # Simpan data yang sudah diproses
    with open('data/processed_data.pkl', 'wb') as f:
        pickle.dump((X, y), f)