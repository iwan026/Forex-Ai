import numpy as np
import pandas as pd
import tensorflow as tf
import talib
import pickle
from data_processor import load_data, add_technical_indicators
import config


def analyze_market_conditions(df):
    """Analisis kondisi pasar dengan TA-Lib"""
    latest = df.iloc[-1]
    conditions = []

    # Analisis Momentum
    rsi = latest["rsi_14"]
    if rsi > 70:
        conditions.append(f"RSI {rsi:.1f} (Overbought)")
    elif rsi < 30:
        conditions.append(f"RSI {rsi:.1f} (Oversold)")

    # Analisis Trend
    adx = latest["adx_14"]
    if adx > 25:
        conditions.append(f"Strong Trend (ADX {adx:.1f})")
    else:
        conditions.append(f"Weak Trend (ADX {adx:.1f})")

    # Analisis MA
    ma_diff = latest["ma_50"] - latest["ma_200"]
    conditions.append(f"MA50/200 Gap: {ma_diff*10000:.1f} pips")

    # Analisis Volatilitas
    atr_pct = (latest["atr_14"] / latest["close"]) * 100
    conditions.append(f"Volatility: {atr_pct:.2f}% (ATR)")

    return conditions


def predict_signal():
    # Load model dengan tf.saved_model untuk kompatibilitas lebih baik
    model = tf.saved_model.load("model/lstm_model")
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load data
    df = load_data("data/EURUSD_H1.csv")
    df = add_technical_indicators(df)

    # Validasi data
    if len(df) < config.LOOKBACK_WINDOW:
        raise ValueError(f"Need at least {config.LOOKBACK_WINDOW} periods")

    # Preprocessing
    latest_data = df[config.FEATURES].tail(config.LOOKBACK_WINDOW)
    scaled_data = scaler.transform(latest_data)

    # Prediksi dengan signature yang jelas
    predict_fn = model.signatures["serving_default"]
    prediction = predict_fn(tf.constant(scaled_data[np.newaxis, ...], dtype=tf.float32))
    probability = prediction["output_0"].numpy()[0][0]

    # Generate signal
    threshold = 0.65  # Lebih konservatif
    signal = "BUY" if probability > threshold else "NO TRADE"

    # Report
    print("\n=== AI Trading Signal ===")
    print(f"Signal: {signal} (Confidence: {probability*100:.1f}%)")
    print(f"Price: {df['close'].iloc[-1]:.5f}")

    # Market analysis
    print("\n=== Market Analysis ===")
    for condition in analyze_market_conditions(df):
        print(f"- {condition}")

    # Key metrics
    print("\n=== Key Metrics ===")
    metrics = {
        "RSI 14": df["rsi_14"].iloc[-1],
        "ATR 14": df["atr_14"].iloc[-1],
        "MACD": df["macd"].iloc[-1],
        "ADX 14": df["adx_14"].iloc[-1],
    }
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")


if __name__ == "__main__":
    predict_signal()
