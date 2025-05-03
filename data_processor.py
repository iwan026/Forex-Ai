import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import RobustScaler
import config
import pickle
import pywt


# data_processor.py (hanya bagian yang relevan)
def add_technical_indicators(df):
    """Menambahkan indikator teknikal menggunakan TA-Lib"""
    df = df.copy()

    # Indikator Momentum
    df["rsi_14"] = talib.RSI(df["close"], timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd, macd_signal, macd_hist
    df["stoch_k_14"], df["stoch_d_14"] = talib.STOCH(
        df["high"],
        df["low"],
        df["close"],
        fastk_period=14,
        slowk_period=3,
        slowd_period=3,
    )
    df["cci_14"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod=14)
    df["mfi_14"] = talib.MFI(
        df["high"], df["low"], df["close"], df["tick_volume"], timeperiod=14
    )

    # Indikator Volatilitas
    df["atr_14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["natr_14"] = talib.NATR(df["high"], df["low"], df["close"], timeperiod=14)
    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
    df["bollinger_upper"], df["bollinger_middle"], df["bollinger_lower"] = (
        upper,
        middle,
        lower,
    )
    df["bb_width"] = (upper - lower) / middle

    # Indikator Trend
    df["ma_50"] = talib.SMA(df["close"], timeperiod=50)
    df["ma_200"] = talib.SMA(df["close"], timeperiod=200)
    df["ema_9"] = talib.EMA(df["close"], timeperiod=9)
    df["ema_21"] = talib.EMA(df["close"], timeperiod=21)
    df["adx_14"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
    df["di_plus_14"] = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=14)
    df["di_minus_14"] = talib.MINUS_DI(
        df["high"], df["low"], df["close"], timeperiod=14
    )
    df["kama_10"] = talib.KAMA(df["close"], timeperiod=10)
    df["adx_trend_strength"] = (
        df["adx_14"] * (df["di_plus_14"] - df["di_minus_14"]).abs()
    )

    # Indikator Volume
    df["obv"] = talib.OBV(df["close"], df["tick_volume"])

    # Fitur tambahan
    df["rsi_normalized"] = df["rsi_14"] / df["atr_14"]
    df["ma_diff"] = (df["ma_50"] - df["ma_200"]) / df["close"]
    df["price_ma50_diff"] = (df["close"] - df["ma_50"]) / df["close"]
    df["price_ma200_diff"] = (df["close"] - df["ma_200"]) / df["close"]

    df.interpolate(method="linear", inplace=True)
    df.bfill(inplace=True)

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
    df = pd.read_csv(filepath, dtype=dtypes, parse_dates=["time"], index_col="time")
    return df


def create_target(df):
    """Target dinamis untuk klasifikasi multi-kelas"""
    df["next_close"] = df["close"].shift(-1)
    pips_change = (df["next_close"] - df["close"]) * 10000
    dynamic_threshold = df["atr_14"] * 10000 * config.ATR_MULTIPLIER

    df[config.TARGET] = np.select(
        [
            (pips_change > dynamic_threshold),
            (pips_change < -dynamic_threshold),
            True,
        ],
        [1, 0, 0.5],  # Pertahankan neutral
        default=0.5,
    )

    return df.dropna()


def preprocess_data(df):
    """Normalisasi data dengan RobustScaler"""
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[config.FEATURES])
    with open(config.SCALER_PATH, "wb") as f:
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
    df = load_data(config.DATA_PATH)
    df = add_technical_indicators(df)
    df = create_target(df)
    processed_data = preprocess_data(df)
    X, y = create_sequences(processed_data)
    with open("data/processed_data.pkl", "wb") as f:
        pickle.dump((X, y), f)
