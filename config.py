# Parameter Model
# 60 candle H1
LOOKBACK_WINDOW = 60
# Fitur dasar
FEATURES = ['open', 'high', 'low', 'close']
# 1=buy, 0=sell
TARGET = 'signal'

# Training
EPOCHS = 50
BATCH_SIZE = 32
# 20% data untuk testing
TEST_SIZE = 0.2