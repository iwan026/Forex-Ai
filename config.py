# Parameter Model
LOOKBACK_WINDOW = 60
RAW_FEATURES = ["open", "high", "low", "close", "tick_volume"]

# TA-Lib indikator
TECHNICAL_INDICATORS = [
    # Momentum indicators
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "stoch_k_14",
    "stoch_d_14",
    "cci_14",
    # Volatility indicators
    "atr_14",
    "natr_14",
    "bollinger_upper",
    "bollinger_middle",
    "bollinger_lower",
    # Trend indicators
    "ma_50",
    "ma_200",
    "ema_9",
    "ema_21",
    "adx_14",
    "di_plus_14",
    "di_minus_14",
    "kama_10",
    # Volume indicators
    "obv",
    "mfi_14",
    # Custom derivatives
    "rsi_normalized",
    "bb_width",
    "adx_trend_strength",
]

FEATURES = RAW_FEATURES + TECHNICAL_INDICATORS
TARGET = "signal"

# Training Parameters
EPOCHS = 200
BATCH_SIZE = 64
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
OPTIMIZER = "adamw"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6

# Hyperparameters
LSTM_UNITS = 256
ATTENTION_HEADS = 8
DROPOUT_RATE = 0.4
L1_REG = 1e-4
L2_REG = 1e-4

# Threshold dinamis berdasarkan ROC
PREDICTION_THRESHOLD = 0.6

# Paths
DATA_PATH = "data/EURUSD_H1.csv"
MODEL_PATH = "model/lstm_model.keras"
SCALER_PATH = "model/scaler.pkl"
BEST_MODEL_PATH = "model/best_model.keras"

# Backtesting parameters
RISK_PER_TRADE = 0.01
ATR_MULTIPLIER = 2.0
