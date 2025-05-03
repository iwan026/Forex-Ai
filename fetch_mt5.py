import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Login ke akun MT5 (opsional jika hanya butuh data)
# mt5.login(login=123456, password="password", server="broker_server")

# Inisialisasi MT5
if not mt5.initialize():
    print("Gagal inisialisasi MT5, error code:", mt5.last_error())
    quit()

# Konfigurasi
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H1  # Timeframe 1 jam
start_date = datetime(2019, 1, 1)
end_date = datetime(2024, 12, 30)

# Ambil data
rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

# Konversi ke DataFrame
df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")  # Konversi Unix time ke datetime
df.set_index("time", inplace=True)

# Simpan ke CSV
df.to_csv(f"{symbol}_H1.csv", columns=["open", "high", "low", "close", "tick_volume"])
print(f"Data disimpan: {symbol}_H1.csv")

# Tutup koneksi
mt5.shutdown()
