import os
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import telebot
import pandas_ta as ta
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from threading import Thread
import configparser
import json
import time
from feature_engineering import ForexFeatureEngineering

# Konfigurasi logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filename="forex_bot.log",
)
logger = logging.getLogger(__name__)

# Konfigurasi dari file config
config = configparser.ConfigParser()
if os.path.exists("config.ini"):
    config.read("config.ini")
else:
    config["TELEGRAM"] = {"BOT_TOKEN": "YOUR_BOT_TOKEN_HERE"}
    config["METATRADER"] = {
        "PATH": r"C:\Program Files\MetaTrader 5\terminal64.exe",
        "LOGIN": "12345",
        "PASSWORD": "password",
        "SERVER": "MetaQuotes-Demo",
    }
    config["MODEL"] = {"MODEL_PATH": "models", "DATA_PATH": "data"}
    with open("config.ini", "w") as f:
        config.write(f)

# Konstanta
TIME_FRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}

# Pastikan direktori untuk model dan data ada
os.makedirs(config["MODEL"]["MODEL_PATH"], exist_ok=True)
os.makedirs(config["MODEL"]["DATA_PATH"], exist_ok=True)

# Inisialisasi bot Telegram
bot = telebot.TeleBot(config["TELEGRAM"]["BOT_TOKEN"])


class ForexPredictor:
    def __init__(self):
        self.mt5_initialized = False
        self.models = {}
        self.scalers = {}
        self.fe = ForexFeatureEngineering(
            denoising=True,
            use_talib=True,
            use_pca=False
        )
        self.initialize_mt5()

    def initialize_mt5(self):
        """Inisialisasi koneksi MT5"""
        if not mt5.initialize(
            path=config["METATRADER"]["PATH"],
            login=int(config["METATRADER"]["LOGIN"]),
            password=config["METATRADER"]["PASSWORD"],
            server=config["METATRADER"]["SERVER"],
        ):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        logger.info("MT5 initialized successfully")
        self.mt5_initialized = True
        return True

    def get_historical_data(self, pair, timeframe, bars=5000):
        """Ambil data historis dari MT5"""
        if not self.mt5_initialized:
            if not self.initialize_mt5():
                return None

        tf = TIME_FRAMES.get(timeframe)
        if not tf:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None

        rates = mt5.copy_rates_from_pos(pair, tf, 0, bars)
        if rates is None or len(rates) == 0:
            logger.error(
                f"Failed to get rates for {pair} {timeframe}: {mt5.last_error()}"
            )
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    def prepare_data(self, df, window_size=60, predict_size=1):
        """Siapkan data untuk input model"""
        # Tambahkan indikator
        processed_df, features = self.fe.process(df)

        # Buat target: arah pergerakan (naik/turun)
        df["target"] = (df["close"].shift(-predict_size) > df["close"]).astype(int)

        # Hapus baris dengan NaN
        df = df.dropna()

        # Pilih data yang akan digunakan
        data = df[features].values
        targets = df["target"].values

        return data, targets, df

    def create_sequences(self, data, targets, window_size=60):
        """Buat sequence untuk input LSTM"""
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(targets[i + window_size])
        return np.array(X), np.array(y)

    def create_model(self, input_shape):
        """Buat model LSTM"""
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def train_model(self, pair, timeframe):
        """Latih model untuk pair dan timeframe tertentu"""
        model_id = f"{pair}_{timeframe}"
        logger.info(f"Training model for {model_id}")

        # Ambil data historis
        df = self.get_historical_data(pair, timeframe, bars=5000)
        if df is None:
            return False

        # Siapkan data
        data, targets, _ = self.prepare_data(df)

        # Normalisasi data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # Buat sequences
        X, y = self.create_sequences(data_scaled, targets)

        # Split train/test
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Buat dan latih model
        model = self.create_model((X.shape[1], X.shape[2]))
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1,
        )

        # Simpan model dan scaler
        model_path = os.path.join(config["MODEL"]["MODEL_PATH"], f"{model_id}.h5")
        scaler_path = os.path.join(
            config["MODEL"]["MODEL_PATH"], f"{model_id}_scaler.pkl"
        )

        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        # Evaluasi model
        test_loss, test_acc = model.evaluate(X_test, y_test)
        logger.info(f"Model {model_id} - Test accuracy: {test_acc:.4f}")

        # Simpan model dan scaler ke cache
        self.models[model_id] = model
        self.scalers[model_id] = scaler

        return True

    def load_model(self, pair, timeframe):
        """Muat model yang sudah ada atau latih jika belum ada"""
        model_id = f"{pair}_{timeframe}"
        model_path = os.path.join(config["MODEL"]["MODEL_PATH"], f"{model_id}.h5")
        scaler_path = os.path.join(
            config["MODEL"]["MODEL_PATH"], f"{model_id}_scaler.pkl"
        )

        # Cek apakah model sudah di cache
        if model_id in self.models and model_id in self.scalers:
            return self.models[model_id], self.scalers[model_id]

        # Cek apakah model sudah ada di disk
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)

                # Simpan ke cache
                self.models[model_id] = model
                self.scalers[model_id] = scaler

                return model, scaler
            except Exception as e:
                logger.error(f"Error loading model: {e}")

        # Latih model baru jika tidak ada
        if self.train_model(pair, timeframe):
            return self.models[model_id], self.scalers[model_id]

        return None, None

    def predict(self, pair, timeframe):
        """Buat prediksi untuk pair dan timeframe"""
        # Muat atau latih model
        model, scaler = self.load_model(pair, timeframe)
        if model is None or scaler is None:
            return None, None, None

        # Ambil data terbaru
        df = self.get_historical_data(pair, timeframe, bars=100)
        if df is None:
            return None, None, None

        # Siapkan data
        data, _, processed_df = self.prepare_data(df)

        # Normalisasi data
        data_scaled = scaler.transform(data)

        # Ambil 60 candle terakhir untuk input
        input_sequence = data_scaled[-60:].reshape(1, 60, data_scaled.shape[1])

        # Buat prediksi
        prediction = model.predict(input_sequence)[0][0]

        # Terjemahkan ke sinyal
        signal = (
            "Buy" if prediction > 0.6 else "Sell" if prediction < 0.4 else "No Trade"
        )

        # Ambil support/resistance dan level psikologis
        support_resistance = self.detect_support_resistance(processed_df)
        psych_levels = self.detect_psychological_levels(processed_df)

        return (
            signal,
            prediction,
            {
                "support_resistance": support_resistance,
                "psychological_levels": psych_levels,
            },
        )

    def backtest(self, pair, timeframe, samples=10):
        """Evaluasi model dengan data historis acak"""
        # Muat atau latih model
        model, scaler = self.load_model(pair, timeframe)
        if model is None or scaler is None:
            return None

        # Ambil data historis lengkap
        df = self.get_historical_data(pair, timeframe, bars=1000)
        if df is None:
            return None

        # Siapkan data
        data, targets, _ = self.prepare_data(df)

        # Normalisasi data
        data_scaled = scaler.transform(data)

        # Buat sequences
        X, y = self.create_sequences(data_scaled, targets)

        # Pilih sampel acak
        total_samples = len(X)
        if samples > total_samples:
            samples = total_samples

        indices = np.random.choice(total_samples, samples, replace=False)
        X_samples = X[indices]
        y_samples = y[indices]

        # Buat prediksi
        predictions = model.predict(X_samples)

        # Hitung akurasi
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        accuracy = (predicted_classes == y_samples).mean()

        # Detail prediksi
        details = []
        for i in range(samples):
            details.append(
                {
                    "actual": int(y_samples[i]),
                    "predicted": float(predictions[i][0]),
                    "correct": int(predicted_classes[i] == y_samples[i]),
                }
            )

        return {"accuracy": float(accuracy), "samples": samples, "details": details}


# Handler untuk command /start
@bot.message_handler(commands=["start"])
def handle_start(message):
    bot.reply_to(
        message,
        "👋 Selamat datang di Bot Prediksi Forex!\n\n"
        "Perintah yang tersedia:\n"
        "/prediksi [PAIR] [TIMEFRAME] - Analisis dan prediksi\n"
        "/train [PAIR] [TIMEFRAME] - Latih model baru\n"
        "/backtest [PAIR] [TIMEFRAME] [SAMPLES] - Uji akurasi model\n\n"
        "Contoh: /prediksi EURUSD M15",
    )


# Handler untuk command /prediksi
@bot.message_handler(commands=["prediksi"])
def handle_predict(message):
    try:
        parts = message.text.split()
        if len(parts) != 3:
            bot.reply_to(
                message,
                "Format yang benar: /prediksi [PAIR] [TIMEFRAME]\nContoh: /prediksi EURUSD M15",
            )
            return

        pair = parts[1].upper()
        timeframe = parts[2].upper()

        if timeframe not in TIME_FRAMES:
            bot.reply_to(
                message,
                f"Timeframe tidak valid. Gunakan salah satu: {', '.join(TIME_FRAMES.keys())}",
            )
            return

        bot.reply_to(
            message, f"Menganalisis {pair} pada timeframe {timeframe}... Mohon tunggu."
        )

        # Buat prediksi dalam thread terpisah
        def make_prediction():
            try:
                predictor = ForexPredictor()
                signal, confidence, analysis = predictor.predict(pair, timeframe)

                if signal is None:
                    bot.send_message(
                        message.chat.id,
                        "Gagal membuat prediksi. Periksa kembali pair dan timeframe.",
                    )
                    return

                # Format output
                confidence_pct = (
                    abs(confidence - 0.5) * 200
                )  # Convert to percentage deviation from 50%

                response = f"🔍 *Analisis {pair} ({timeframe})*\n\n"
                response += f"📊 *Sinyal:* {signal}\n"
                response += f"🎯 *Keyakinan:* {confidence_pct:.1f}%\n\n"

                # Tambahkan support/resistance
                if analysis["support_resistance"]:
                    response += "*Level Support/Resistance:*\n"
                    sr_levels = sorted(analysis["support_resistance"].keys())
                    for i, level in enumerate(sr_levels[:3]):  # Tampilkan max 3 level
                        response += f"• {level:.5f}\n"

                # Tambahkan level psikologis
                if analysis["psychological_levels"]:
                    response += "\n*Level Psikologis:*\n"
                    for i, (level, count) in enumerate(
                        analysis["psychological_levels"][:3]
                    ):  # Tampilkan max 3 level
                        response += f"• {level:.5f} ({count} interaksi)\n"

                bot.send_message(message.chat.id, response, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Error in prediction thread: {e}")
                bot.send_message(message.chat.id, f"Terjadi kesalahan: {str(e)}")

        Thread(target=make_prediction).start()
    except Exception as e:
        logger.error(f"Error in handle_predict: {e}")
        bot.reply_to(message, f"Terjadi kesalahan: {str(e)}")


# Handler untuk command /train
@bot.message_handler(commands=["train"])
def handle_train(message):
    try:
        parts = message.text.split()
        if len(parts) != 3:
            bot.reply_to(
                message,
                "Format yang benar: /train [PAIR] [TIMEFRAME]\nContoh: /train EURUSD M15",
            )
            return

        pair = parts[1].upper()
        timeframe = parts[2].upper()

        if timeframe not in TIME_FRAMES:
            bot.reply_to(
                message,
                f"Timeframe tidak valid. Gunakan salah satu: {', '.join(TIME_FRAMES.keys())}",
            )
            return

        bot.reply_to(
            message,
            f"Melatih model untuk {pair} pada timeframe {timeframe}... Ini akan memakan waktu beberapa menit.",
        )

        # Latih model dalam thread terpisah
        def train_model():
            try:
                predictor = ForexPredictor()
                success = predictor.train_model(pair, timeframe)

                if success:
                    bot.send_message(
                        message.chat.id,
                        f"✅ Model untuk {pair} ({timeframe}) berhasil dilatih!",
                    )
                else:
                    bot.send_message(
                        message.chat.id,
                        f"❌ Gagal melatih model. Periksa kembali pair dan timeframe.",
                    )
            except Exception as e:
                logger.error(f"Error in training thread: {e}")
                bot.send_message(message.chat.id, f"Terjadi kesalahan: {str(e)}")

        Thread(target=train_model).start()
    except Exception as e:
        logger.error(f"Error in handle_train: {e}")
        bot.reply_to(message, f"Terjadi kesalahan: {str(e)}")


# Handler untuk command /backtest
@bot.message_handler(commands=["backtest"])
def handle_backtest(message):
    try:
        parts = message.text.split()
        if len(parts) < 3:
            bot.reply_to(
                message,
                "Format yang benar: /backtest [PAIR] [TIMEFRAME] [SAMPLES]\nContoh: /backtest EURUSD M15 20",
            )
            return

        pair = parts[1].upper()
        timeframe = parts[2].upper()
        samples = 10  # Default
        if len(parts) > 3:
            try:
                samples = int(parts[3])
            except:
                pass

        if timeframe not in TIME_FRAMES:
            bot.reply_to(
                message,
                f"Timeframe tidak valid. Gunakan salah satu: {', '.join(TIME_FRAMES.keys())}",
            )
            return

        bot.reply_to(
            message,
            f"Menjalankan backtest untuk {pair} pada timeframe {timeframe} dengan {samples} sampel... Mohon tunggu.",
        )

        # Jalankan backtest dalam thread terpisah
        def run_backtest():
            try:
                predictor = ForexPredictor()
                results = predictor.backtest(pair, timeframe, samples)

                if results is None:
                    bot.send_message(
                        message.chat.id,
                        "Gagal menjalankan backtest. Periksa kembali pair dan timeframe.",
                    )
                    return

                # Format output
                response = f"📈 *Hasil Backtest {pair} ({timeframe})*\n\n"
                response += f"🎯 *Akurasi:* {results['accuracy'] * 100:.2f}%\n"
                response += f"🔢 *Jumlah Sampel:* {results['samples']}\n\n"

                # Detail sampel
                correct_count = sum(d["correct"] for d in results["details"])
                response += f"✅ *Benar:* {correct_count}\n"
                response += f"❌ *Salah:* {results['samples'] - correct_count}\n"

                bot.send_message(message.chat.id, response, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Error in backtest thread: {e}")
                bot.send_message(message.chat.id, f"Terjadi kesalahan: {str(e)}")

        Thread(target=run_backtest).start()
    except Exception as e:
        logger.error(f"Error in handle_backtest: {e}")
        bot.reply_to(message, f"Terjadi kesalahan: {str(e)}")


if __name__ == "__main__":
    # Pastikan MT5 bisa terkoneksi
    predictor = ForexPredictor()
    if not predictor.mt5_initialized:
        logger.error("MT5 failed to initialize. Please check the configuration.")

    # Jalankan bot
    logger.info("Starting bot...")
    bot.polling(none_stop=True)
