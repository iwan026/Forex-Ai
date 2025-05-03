import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Inisialisasi MT5
if not mt5.initialize():
    print("Gagal inisialisasi MT5, error code:", mt5.last_error())
    quit()

# Konfigurasi
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H1
start_date = datetime(2023, 1, 1)
end_date = datetime.now()

# Ambil data
rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

# Konversi ke DataFrame
df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s") 
df.set_index("time", inplace=True)

# Simpan ke CSV
df.to_csv(f"{symbol}_H1.csv", columns=["open", "high", "low", "close", "volume"])
print(f"Data disimpan: {symbol}_H1.csv")

# Tutup koneksi
mt5.shutdown()
