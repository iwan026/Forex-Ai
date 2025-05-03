# predictor.py
import numpy as np
import pandas as pd
import tensorflow as tf
from data_processor import load_data, add_technical_indicators
import config


def analyze_market_conditions(df):
    """Analisis kondisi pasar dengan TA-Lib"""
    latest = df.iloc[-1]
    conditions = []

    # Analisis Momentum
    rsi = latest["rsi_14"]
    conditions.append(
        f"RSI {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})"
    )

    # Analisis Trend
    adx = latest["adx_14"]
    conditions.append(f"{'Strong' if adx > 25 else 'Weak'} Trend (ADX {adx:.1f})")

    # Analisis MA
    ma_diff = latest["ma_50"] - latest["ma_200"]
    conditions.append(f"MA50/200 Gap: {ma_diff*10000:.1f} pips")

    # Analisis Volatilitas
    atr_pct = (latest["atr_14"] / latest["close"]) * 100
    conditions.append(f"Volatility: {atr_pct:.2f}% (ATR)")

    # Analisis KAMA
    kama_diff = latest["close"] - latest["kama_10"]
    conditions.append(f"KAMA Diff: {kama_diff*10000:.1f} pips")

    return conditions


def predict_signal():
    model = tf.keras.models.load_model(
        config.MODEL_PATH, custom_objects={"focal_loss_fn": None}
    )
    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load data
    df = load_data(config.DATA_PATH)
    df = add_technical_indicators(df)

    if len(df) < config.LOOKBACK_WINDOW:
        raise ValueError(f"Need at least {config.LOOKBACK_WINDOW} periods")

    # Preprocessing untuk ensemble
    latest_data = df[config.FEATURES].tail(config.LOOKBACK_WINDOW + 10)
    predictions = []
    for i in range(10):
        window = latest_data.iloc[i : i + config.LOOKBACK_WINDOW]
        scaled_data = scaler.transform(window)
        pred = model.predict(scaled_data[np.newaxis, ...], verbose=0)
        predictions.append(pred[0])

    # Rata-rata prediksi
    probability = np.mean(predictions, axis=0)
    signal = np.argmax(probability)
    signal_map = {0: "SELL", 1: "BUY", 2: "NEUTRAL"}
    signal_text = signal_map[signal]

    # Report
    print("\n=== AI Trading Signal ===")
    print(f"Signal: {signal_text} (Confidence: {probability[signal]*100:.1f}%)")
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
        "KAMA 10": df["kama_10"].iloc[-1],
    }
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")


if __name__ == "__main__":
    predict_signal()
