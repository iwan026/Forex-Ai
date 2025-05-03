# Parameter Model
LOOKBACK_WINDOW = 60
RAW_FEATURES = ["open", "high", "low", "close", "tick_volume"]

# Fitur TA-Lib yang ditingkatkan dengan indikator tambahan
TECHNICAL_INDICATORS = [
    # Momentum indicators
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",  # Ditambahkan histogram MACD
    "stoch_k_14",  # Stochastic oscillator
    "stoch_d_14",
    "cci_14",  # Commodity Channel Index
    
    # Volatility indicators
    "atr_14",
    "natr_14",
    "bollinger_upper",  # Bollinger Bands
    "bollinger_middle",
    "bollinger_lower",
    
    # Trend indicators
    "ma_50",
    "ma_200",
    "ema_9",   # Exponential Moving Average
    "ema_21",
    "adx_14",
    "di_plus_14",  # Directional indicators
    "di_minus_14",
    
    # Volume indicators
    "obv",
    "mfi_14",  # Money Flow Index
    
    # Custom derivatives
    "ma_diff",
    "price_ma50_diff",
    "price_ma200_diff",
    "bb_width",  # Bollinger bandwidth
    "adx_trend_strength",  # ADX & DI combination
]

FEATURES = RAW_FEATURES + TECHNICAL_INDICATORS

TARGET = "signal"

# Training Parameters yang dioptimalkan
EPOCHS = 150  # Ditingkatkan untuk learning yang lebih baik
BATCH_SIZE = 32  # Ukuran batch yang lebih kecil untuk generalisasi lebih baik
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # Ditambahkan validation split terpisah
OPTIMIZER = "adamw"
LEARNING_RATE = 3e-4  # Ditingkatkan sedikit untuk konvergensi yang lebih baik
WEIGHT_DECAY = 1e-5   # L2 regularization dalam AdamW

# Hyperparameters tambahan untuk model
LSTM_UNITS = 128
ATTENTION_HEADS = 4
DROPOUT_RATE = 0.3
L1_REG = 0.001
L2_REG = 0.001

# Threshold untuk keputusan trading
PREDICTION_THRESHOLD = 0.65  # Untuk keputusan BUY

# Paths
DATA_PATH = "data/EURUSD_H1.csv"
MODEL_PATH = "model/lstm_model"
SCALER_PATH = "model/scaler.pkl"
BEST_MODEL_PATH = "model/best_model"

# Backtesting parameters
RISK_PER_TRADE = 0.01  # 1% dari kapital
ATR_MULTIPLIER = 1.5   # Untuk menentukan stop loss