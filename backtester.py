import numpy as np
import pandas as pd
import tensorflow as tf
import talib
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, RocCurveDisplay
from data_processor import load_data, add_technical_indicators, create_target
import config

def calculate_trading_metrics(results):
    """Metrik trading yang lebih komprehensif"""
    results['next_close'] = results['close'].shift(-1)
    
    # Dynamic position sizing based on volatility
    risk_per_trade = 0.01  # 1% of capital
    results['position_size'] = risk_per_trade / (results['atr_14'] * 1.5)
    
    # Calculate returns
    results['pips_gain'] = np.where(
        results['predicted'] == 1,
        (results['next_close'] - results['close']) * 10000,
        0
    )
    
    # Risk-adjusted metrics
    results['risk_adj_return'] = results['pips_gain'] / results['atr_14']
    
    # Trade classification
    results['trade_quality'] = np.select(
        [
            (results['predicted'] == 1) & (results['actual'] == 1),
            (results['predicted'] == 1) & (results['actual'] == 0),
            (results['predicted'] == 0) & (results['actual'] == 1),
            (results['predicted'] == 0) & (results['actual'] == 0)
        ],
        ['win', 'loss', 'missed', 'correct_reject'],
        default='unknown'
    )
    
    # Advanced metrics
    trades = results[results['predicted'] == 1]
    if len(trades) > 0:
        win_rate = len(trades[trades['trade_quality'] == 'win']) / len(trades)
        avg_win = trades[trades['trade_quality'] == 'win']['pips_gain'].mean()
        avg_loss = trades[trades['trade_quality'] == 'loss']['pips_gain'].mean()
        profit_factor = (win_rate * avg_win) / ((1-win_rate) * abs(avg_loss))
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0
    
    print("\n=== Advanced Metrics ===")
    print(f"Win Rate:          {win_rate*100:.1f}%")
    print(f"Avg Win/Loss:      {avg_win:.1f}/{avg_loss:.1f} pips")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Total Pips:        {results['pips_gain'].sum():.1f}")
    
    return results

def plot_backtest(results):
    """Visualisasi canggih dengan subplots"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), 
                                      gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Price plot
    ax1.plot(results['time'], results['close'], label='Price')
    ax1.plot(results['time'], results['ma_50'], label='MA 50', alpha=0.5)
    ax1.plot(results['time'], results['ma_200'], label='MA 200', alpha=0.5)
    
    # Mark trades
    wins = results[results['trade_quality'] == 'win']
    losses = results[results['trade_quality'] == 'loss']
    ax1.scatter(wins['time'], wins['close'], color='green', label='Win', marker='^')
    ax1.scatter(losses['time'], losses['close'], color='red'], label='Loss', marker='v')
    ax1.legend()
    
    # Indicator plots
    ax2.plot(results['time'], results['rsi_14'], label='RSI 14', color='purple')
    ax2.axhline(70, color='red'], linestyle='--')
    ax2.axhline(30, color='green'], linestyle='--')
    
    ax3.plot(results['time'], results['atr_14'], label='ATR 14', color='orange')
    
    plt.tight_layout()
    plt.savefig('advanced_backtest.png', dpi=300)
    plt.show()

def backtest_model():
    # Load model dengan saved_model format
    model = tf.saved_model.load("model/lstm_model")
    predict_fn = model.signatures["serving_default"]
    
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Load dan proses data
    df = load_data("data/EURUSD_H1.csv")
    df = add_technical_indicators(df)
    df = create_target(df)
    
    # Preprocessing
    scaled_data = scaler.transform(df[config.FEATURES])
    X, y_true = [], []
    for i in range(config.LOOKBACK_WINDOW, len(scaled_data)):
        X.append(scaled_data[i-config.LOOKBACK_WINDOW:i])
        y_true.append(df[config.TARGET].iloc[i-1])
    
    # Prediksi
    y_pred_prob = np.array([
        predict_fn(tf.constant(x[np.newaxis, ...], dtype=tf.float32))['output_0'].numpy()[0][0]
        for x in X
    ])
    
    # Threshold optimal
    RocCurveDisplay.from_predictions(y_true, y_pred_prob)
    plt.savefig('roc_curve.png')
    
    # Simpan hasil
    results = df.iloc[config.LOOKBACK_WINDOW:].copy()
    results['predicted'] = (y_pred_prob > 0.5).astype(int)
    results['probability'] = y_pred_prob
    results['actual'] = y_true
    
    # Evaluasi
    print(classification_report(y_true, results['predicted']))
    
    results = calculate_trading_metrics(results)
    plot_backtest(results)
    
    return results

if __name__ == "__main__":
    backtest_results = backtest_model()