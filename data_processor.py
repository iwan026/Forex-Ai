import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
import config
import pickle


def add_technical_indicators(df):
    """Menambahkan indikator teknikal menggunakan TA-Lib"""
    df = df.copy()

    # Indikator Momentum
    df["rsi_14"] = talib.RSI(df["close"], timeperiod=14)
    df["macd"], df["macd_signal"], _ = talib.MACD(df["close"])

    # Indikator Volatilitas
    df["atr_14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["natr_14"] = talib.NATR(df["high"], df["low"], df["close"], timeperiod=14)

    # Indikator Trend
    df["ma_50"] = talib.SMA(df["close"], timeperiod=50)
    df["ma_200"] = talib.SMA(df["close"], timeperiod=200)
    df["adx_14"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)

    # Indikator Volume
    df["obv"] = talib.OBV(df["close"], df["tick_volume"])

    # Fitur tambahan
    df["ma_diff"] = (df["ma_50"] - df["ma_200"]) / df["close"]
    df["price_ma50_diff"] = (df["close"] - df["ma_50"]) / df["close"]
    df["price_ma200_diff"] = (df["close"] - df["ma_200"]) / df["close"]

    # Handle NaN values
    df.fillna(method="bfill", inplace=True)

    return df


def load_data(filepath):
    """Load data dengan optimasi memori"""
    dtypes = {
        "open": "float32",
        "high": "float32",
        "low": "float32",
        "close": "float32",
        "tick_volume": "float32",
    }
    df = pd.read_csv(filepath, dtype=dtypes)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df


def create_target(df):
    """Target dinamis berbasis ATR dan kondisi pasar"""
    df["next_close"] = df["close"].shift(-1)
    pips_change = (df["next_close"] - df["close"]) * 10000

    # Threshold dinamis berdasarkan ATR
    atr_multiplier = 1.5
    dynamic_threshold = df["atr_14"] * 10000 * atr_multiplier

    # Kondisi tambahan berdasarkan indikator
    rsi_condition = (df["rsi_14"] < 30) | (df["rsi_14"] > 70)
    ma_condition = df["ma_50"] > df["ma_200"]

    df[config.TARGET] = np.where(
        (pips_change > dynamic_threshold) & rsi_condition & ma_condition,
        1,
        np.where(
            (pips_change < -dynamic_threshold) & rsi_condition & (~ma_condition),
            0,
            0.5,  # Neutral
        ),
    )

    # Hapus sinyal neutral
    df = df[df[config.TARGET] != 0.5]
    df[config.TARGET] = df[config.TARGET].astype(int)

    return df.dropna()


def preprocess_data(df):
    """Normalisasi data dan simpan scaler"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[config.FEATURES])
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return scaled_data


def create_sequences(data):
    """Buat sequences untuk LSTM"""
    X, y = [], []
    for i in range(config.LOOKBACK_WINDOW, len(data)):
        X.append(data[i - config.LOOKBACK_WINDOW : i])
        y.append(data[i - 1, -1])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    df = load_data("data/EURUSD_H1.csv")
    df = add_technical_indicators(df)
    df = create_target(df)
    processed_data = preprocess_data(df)
    X, y = create_sequences(processed_data)
    with open("data/processed_data.pkl", "wb") as f:
        pickle.dump((X, y), f)
