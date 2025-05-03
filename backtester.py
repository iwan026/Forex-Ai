# backtester.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, RocCurveDisplay
from data_processor import load_data, add_technical_indicators, create_target
import config


def calculate_trading_metrics(results):
    """Metrik trading dengan Sharpe dan Sortino Ratio"""
    results["next_close"] = results["close"].shift(-1)
    results["position_size"] = config.RISK_PER_TRADE / (
        results["atr_14"] * config.ATR_MULTIPLIER
    )
    results["pips_gain"] = np.where(
        results["predicted"] == 1,
        (results["next_close"] - results["close"]) * 10000,
        np.where(
            results["predicted"] == 0,
            (results["close"] - results["next_close"]) * 10000,
            0,
        ),
    )
    results["risk_adj_return"] = results["pips_gain"] / results["atr_14"]

    # Equity curve
    results["equity"] = results["pips_gain"].cumsum()

    # Trade classification
    results["trade_quality"] = np.select(
        [
            (results["predicted"] == results["actual"]),
            (results["predicted"] != results["actual"]),
        ],
        ["win", "loss"],
        default="unknown",
    )

    # Metrik
    trades = results[results["predicted"] != 2]  # Exclude neutral
    if len(trades) > 0:
        win_rate = len(trades[trades["trade_quality"] == "win"]) / len(trades)
        avg_win = trades[trades["trade_quality"] == "win"]["pips_gain"].mean()
        avg_loss = trades[trades["trade_quality"] == "loss"]["pips_gain"].mean()
        profit_factor = (
            (win_rate * avg_win) / ((1 - win_rate) * abs(avg_loss))
            if avg_loss != 0
            else np.inf
        )
        sharpe_ratio = (
            results["pips_gain"].mean() / results["pips_gain"].std() * np.sqrt(252)
            if results["pips_gain"].std() != 0
            else 0
        )
        sortino_ratio = (
            results["pips_gain"].mean()
            / results[results["pips_gain"] < 0]["pips_gain"].std()
            * np.sqrt(252)
            if results[results["pips_gain"] < 0]["pips_gain"].std() != 0
            else 0
        )
    else:
        win_rate = avg_win = avg_loss = profit_factor = sharpe_ratio = sortino_ratio = 0

    print("\n=== Advanced Metrics ===")
    print(f"Win Rate:          {win_rate*100:.1f}%")
    print(f"Avg Win/Loss:      {avg_win:.1f}/{avg_loss:.1f} pips")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Total Pips:        {results['pips_gain'].sum():.1f}")
    print(f"Sharpe Ratio:      {sharpe_ratio:.2f}")
    print(f"Sortino Ratio:     {sortino_ratio:.2f}")

    return results


def plot_backtest(results):
    """Visualisasi dengan equity curve"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(20, 20), gridspec_kw={"height_ratios": [3, 1, 1, 1]}
    )

    # Price plot
    ax1.plot(results["time"], results["close"], label="Price")
    ax1.plot(results["time"], results["ma_50"], label="MA 50", alpha=0.5)
    ax1.plot(results["time"], results["ma_200"], label="MA 200", alpha=0.5)
    wins = results[results["trade_quality"] == "win"]
    losses = results[results["trade_quality"] == "loss"]
    ax1.scatter(wins["time"], wins["close"], color="green", label="Win", marker="^")
    ax1.scatter(losses["time"], losses["close"], color="red", label="Loss", marker="v")
    ax1.legend()

    # Indicator plots
    ax2.plot(results["time"], results["rsi_14"], label="RSI 14", color="purple")
    ax2.axhline(70, color="red", linestyle="--")
    ax2.axhline(30, color="green", linestyle="--")

    ax3.plot(results["time"], results["atr_14"], label="ATR 14", color="orange")
    ax3.legend()

    # Equity curve
    ax4.plot(results["time"], results["equity"], label="Equity Curve", color="blue")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("advanced_backtest.png", dpi=300)
    plt.show()


def backtest_model():
    model = tf.keras.models.load_model(
        config.MODEL_PATH, custom_objects={"focal_loss_fn": None}
    )
    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load dan proses data
    df = load_data(config.DATA_PATH)
    df = add_technical_indicators(df)
    df = create_target(df)

    # Preprocessing
    scaled_data = scaler.transform(df[config.FEATURES])
    X, y_true = [], []
    for i in range(config.LOOKBACK_WINDOW, len(scaled_data)):
        X.append(scaled_data[i - config.LOOKBACK_WINDOW : i])
        y_true.append(df[config.TARGET].iloc[i - 1])

    # Prediksi
    y_pred_prob = model.predict(np.array(X), verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Optimasi threshold (contoh sederhana)
    RocCurveDisplay.from_predictions(y_true, y_pred_prob[:, 1])
    plt.savefig("roc_curve.png")

    # Simpan hasil
    results = df.iloc[config.LOOKBACK_WINDOW :].copy()
    results["predicted"] = y_pred
    results["probability"] = y_pred_prob.max(axis=1)
    results["actual"] = y_true

    # Evaluasi
    print(
        classification_report(y_true, y_pred, target_names=["SELL", "BUY", "NEUTRAL"])
    )

    results = calculate_trading_metrics(results)
    plot_backtest(results)

    return results


if __name__ == "__main__":
    backtest_results = backtest_model()
