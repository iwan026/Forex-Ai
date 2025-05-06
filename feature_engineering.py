import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import talib


class ForexFeatureEngineering:
    def __init__(
        self, denoising=True, use_talib=True, use_pca=False, pca_components=10
    ):
        """Inisialisasi parameter feature engineering"""
        self.denoising = denoising
        self.use_talib = use_talib
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None
        self.scaler = StandardScaler()

    def denoise_data(self, df):
        """Melakukan denoising pada data harga dengan Savitzky-Golay filter"""
        if not self.denoising:
            return df

        window_length = min(15, len(df) // 4)  # Make sure window length is odd
        if window_length % 2 == 0:
            window_length += 1

        polyorder = min(3, window_length - 1)

        # Apply Savitzky-Golay filter to price data
        try:
            df["open_smooth"] = savgol_filter(df["open"], window_length, polyorder)
            df["high_smooth"] = savgol_filter(df["high"], window_length, polyorder)
            df["low_smooth"] = savgol_filter(df["low"], window_length, polyorder)
            df["close_smooth"] = savgol_filter(df["close"], window_length, polyorder)

            # Replace original columns for model input
            df["open"] = df["open_smooth"]
            df["high"] = df["high_smooth"]
            df["low"] = df["low_smooth"]
            df["close"] = df["close_smooth"]

            # Drop intermediate columns
            df = df.drop(
                ["open_smooth", "high_smooth", "low_smooth", "close_smooth"], axis=1
            )
        except Exception as e:
            print(f"Warning: Denoising failed: {e}. Using original data.")

        return df

    def add_price_features(self, df):
        """Menambahkan fitur-fitur yang terkait dengan harga"""
        # Price differences at various lags
        for lag in [1, 2, 3, 5, 8, 13]:  # Fibonacci sequence
            df[f"close_diff_{lag}"] = df["close"].diff(lag)
            df[f"close_pct_{lag}"] = df["close"].pct_change(lag)

        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Price volatility features
        df["volatility_daily"] = df["log_return"].rolling(window=20).std()
        df["volatility_weekly"] = df["log_return"].rolling(window=5).std()

        # High-Low range features
        df["hl_ratio"] = df["high"] / df["low"]
        df["oc_ratio"] = df["open"] / df["close"]
        df["hl_pct"] = (df["high"] - df["low"]) / df["low"] * 100

        # Moving averages crossover features
        df["ma_cross_5_20"] = (
            df["close"].rolling(5).mean() > df["close"].rolling(20).mean()
        ).astype(int)
        df["ma_cross_10_50"] = (
            df["close"].rolling(10).mean() > df["close"].rolling(50).mean()
        ).astype(int)

        # Price momentum
        df["momentum_5"] = df["close"] - df["close"].shift(5)
        df["momentum_10"] = df["close"] - df["close"].shift(10)
        df["momentum_20"] = df["close"] - df["close"].shift(20)

        # Moving average convergence/divergence rate
        df["mac_rate_5_20"] = (
            df["close"].rolling(5).mean() - df["close"].rolling(20).mean()
        ) / df["close"].rolling(20).mean()

        # Mean reversion features
        df["mean_reversion_20"] = df["close"] / df["close"].rolling(20).mean() - 1
        df["mean_reversion_50"] = df["close"] / df["close"].rolling(50).mean() - 1

        # Candle patterns as numerical features
        df["bullish_candle"] = (df["close"] > df["open"]).astype(int)
        df["doji"] = (
            abs(df["close"] - df["open"]) < 0.1 * (df["high"] - df["low"])
        ).astype(int)
        df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (
            df["high"] - df["low"]
        )
        df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (
            df["high"] - df["low"]
        )

        # Gap features
        df["gap_up"] = (df["open"] > df["close"].shift(1)).astype(int)
        df["gap_down"] = (df["open"] < df["close"].shift(1)).astype(int)
        df["gap_size"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        return df

    def add_pandas_ta_indicators(self, df):
        """Menambahkan indikator teknikal menggunakan pandas-ta"""
        # RSI with different periods
        df["rsi_14"] = ta.rsi(df["close"], length=14)
        df["rsi_7"] = ta.rsi(df["close"], length=7)
        df["rsi_21"] = ta.rsi(df["close"], length=21)

        # MACD
        macd = ta.macd(df["close"])
        df = pd.concat([df, macd], axis=1)

        # Bollinger Bands
        bb = ta.bbands(df["close"], length=20)
        df = pd.concat([df, bb], axis=1)

        # Extra BB feature - position within bands
        df["bb_position"] = (df["close"] - df["BBL_20_2.0"]) / (
            df["BBU_20_2.0"] - df["BBL_20_2.0"]
        )

        # Keltner Channels
        kc = ta.kc(df["high"], df["low"], df["close"])
        df = pd.concat([df, kc], axis=1)

        # Squeeze momentum
        squeeze = ta.squeeze(df["high"], df["low"], df["close"])
        df = pd.concat([df, squeeze], axis=1)

        # EMAs
        for period in [9, 21, 50, 100, 200]:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)

        # ATR and normalized ATR
        df["atr"] = ta.atr(df["high"], df["low"], df["close"])
        df["natr"] = df["atr"] / df["close"] * 100  # Normalized ATR

        # Stochastic
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        df = pd.concat([df, stoch], axis=1)

        # Volume indicators
        if "tick_volume" in df.columns:
            df["vol_roc"] = ta.roc(df["tick_volume"], length=1)
            df["vol_sma_ratio"] = df["tick_volume"] / ta.sma(
                df["tick_volume"], length=20
            )
            df["vol_ema_ratio"] = df["tick_volume"] / ta.ema(
                df["tick_volume"], length=20
            )

            # On-balance Volume
            df["obv"] = ta.obv(df["close"], df["tick_volume"])

            # Money Flow Index
            df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["tick_volume"])

            # Chaikin Money Flow
            df["cmf"] = ta.cmf(df["high"], df["low"], df["close"], df["tick_volume"])

            # Volume-weighted price
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["tick_volume"])

        # Ichimoku Cloud
        ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
        df = pd.concat([df, ichimoku[0]], axis=1)  # Only take the primary dataframe

        # PSAR (Parabolic SAR)
        df["psar"] = ta.psar(df["high"], df["low"])

        # Momentum indicators
        df["cci"] = ta.cci(df["high"], df["low"], df["close"])
        df["adx"] = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
        df["roc"] = ta.roc(df["close"], length=9)

        return df

    def add_talib_indicators(self, df):
        """Menambahkan indikator teknikal menggunakan TALib"""
        if not self.use_talib:
            return df

        try:
            # Pattern recognition
            # Bullish patterns
            df["cdl_hammer"] = talib.CDLHAMMER(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )
            df["cdl_inverted_hammer"] = talib.CDLINVERTEDHAMMER(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )
            df["cdl_engulfing"] = talib.CDLENGULFING(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )
            df["cdl_morning_star"] = talib.CDLMORNINGSTAR(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )

            # Bearish patterns
            df["cdl_hanging_man"] = talib.CDLHANGINGMAN(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )
            df["cdl_shooting_star"] = talib.CDLSHOOTINGSTAR(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )
            df["cdl_evening_star"] = talib.CDLEVENINGSTAR(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )

            # Other patterns
            df["cdl_doji"] = talib.CDLDOJI(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )
            df["cdl_3_outside"] = talib.CDL3OUTSIDE(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )

            # Additional indicators
            df["bop"] = talib.BOP(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )
            df["willr"] = talib.WILLR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
            df["ultosc"] = talib.ULTOSC(
                df["high"].values, df["low"].values, df["close"].values
            )
            df["adx"] = talib.ADX(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )

            # Hilbert Transform
            df["ht_trendline"] = talib.HT_TRENDLINE(df["close"].values)
            df["ht_sine"], df["ht_leadsine"] = talib.HT_SINE(df["close"].values)

            # Normalized indicators
            df["willr_norm"] = (df["willr"] + 100) / 100  # Normalize to 0-1

        except Exception as e:
            print(
                f"Warning: TALib indicators failed: {e}. Using only pandas-ta indicators."
            )

        return df

    def add_support_resistance_features(
        self, df, support_resistance, psychological_levels
    ):
        """Menambahkan fitur yang terkait dengan level support/resistance"""
        current_price = df["close"].iloc[-1]

        # Calculate distance to support/resistance levels
        if support_resistance:
            sr_prices = list(support_resistance.keys())

            # Find closest support (price below current)
            supports = [p for p in sr_prices if p < current_price]
            if supports:
                nearest_support = max(supports)
                df["dist_to_support"] = (df["close"] - nearest_support) / df["close"]
                df["dist_to_support_atr"] = (df["close"] - nearest_support) / df["atr"]
            else:
                df["dist_to_support"] = -1
                df["dist_to_support_atr"] = -1

            # Find closest resistance (price above current)
            resistances = [p for p in sr_prices if p > current_price]
            if resistances:
                nearest_resistance = min(resistances)
                df["dist_to_resistance"] = (nearest_resistance - df["close"]) / df[
                    "close"
                ]
                df["dist_to_resistance_atr"] = (nearest_resistance - df["close"]) / df[
                    "atr"
                ]
            else:
                df["dist_to_resistance"] = -1
                df["dist_to_resistance_atr"] = -1

            # Support/resistance density
            price_range = df["high"].max() - df["low"].min()
            sr_density = len(sr_prices) / price_range if price_range > 0 else 0
            df["sr_density"] = sr_density

            # Ratio of support to resistance
            support_count = len(supports)
            resistance_count = len(resistances)
            total_count = support_count + resistance_count
            df["support_ratio"] = (
                support_count / total_count if total_count > 0 else 0.5
            )
        else:
            df["dist_to_support"] = -1
            df["dist_to_support_atr"] = -1
            df["dist_to_resistance"] = -1
            df["dist_to_resistance_atr"] = -1
            df["sr_density"] = 0
            df["support_ratio"] = 0.5

        # Calculate distance to psychological levels
        if psychological_levels:
            psych_prices = [level for level, _ in psychological_levels]

            # Find closest psychological level
            if psych_prices:
                nearest_psych = min(psych_prices, key=lambda x: abs(x - current_price))
                df["dist_to_psych"] = abs(df["close"] - nearest_psych) / df["close"]

                # Is price approaching a psychological level?
                df["approaching_psych"] = (
                    abs(df["close"] - nearest_psych) < df["atr"]
                ).astype(int)

                # Strength of psychological level (based on interaction count)
                nearest_psych_interactions = [
                    count
                    for level, count in psychological_levels
                    if level == nearest_psych
                ][0]
                df["psych_strength"] = nearest_psych_interactions / max(
                    [count for _, count in psychological_levels]
                )
            else:
                df["dist_to_psych"] = -1
                df["approaching_psych"] = 0
                df["psych_strength"] = 0
        else:
            df["dist_to_psych"] = -1
            df["approaching_psych"] = 0
            df["psych_strength"] = 0

        return df

    def add_market_regime_features(self, df):
        """Menambahkan fitur yang mendeteksi regime pasar (trending/ranging)"""
        # ADX as trend strength indicator
        if "adx" not in df.columns:
            try:
                df["adx"] = talib.ADX(
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                    timeperiod=14,
                )
            except:
                df["adx"] = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]

        # Market regime based on ADX
        df["trending_market"] = (df["adx"] > 25).astype(int)
        df["strong_trend"] = (df["adx"] > 40).astype(int)

        # Volatility based regime
        df["high_volatility"] = (df["atr"] > df["atr"].rolling(20).mean()).astype(int)

        # Determine if we're in a ranging market (low volatility + low ADX)
        df["ranging_market"] = (
            (df["adx"] < 20) & (df["atr"] < df["atr"].rolling(20).mean())
        ).astype(int)

        # Volatility expansion/contraction
        df["volatility_expansion"] = (df["atr"] > df["atr"].shift(1)).astype(int)

        # Detect consolidation periods
        bb_width = (df["BBU_20_2.0"] - df["BBL_20_2.0"]) / df["BBM_20_2.0"]
        df["bb_squeeze"] = (bb_width < bb_width.rolling(20).mean()).astype(int)

        # Detect breakouts
        df["breakout_up"] = (
            (df["close"] > df["BBU_20_2.0"])
            & (df["close"].shift(1) <= df["BBU_20_2.0"].shift(1))
        ).astype(int)
        df["breakout_down"] = (
            (df["close"] < df["BBL_20_2.0"])
            & (df["close"].shift(1) >= df["BBL_20_2.0"].shift(1))
        ).astype(int)

        return df

    def add_time_features(self, df):
        """Menambahkan fitur terkait waktu"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Extract time components
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

        # Market session indicators (approximate UTC times)
        df["asian_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
        df["london_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
        df["ny_session"] = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)
        df["overlap_session"] = ((df["hour"] >= 13) & (df["hour"] < 16)).astype(int)

        # Day features
        df["monday"] = (df["day_of_week"] == 0).astype(int)
        df["friday"] = (df["day_of_week"] == 4).astype(int)

        # Hour of day - cyclic features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

        # Week of year - cyclic features
        if hasattr(df.index, "isocalendar"):
            df["week"] = df.index.isocalendar().week
            df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52.0)
            df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52.0)

        return df

    def add_custom_features(self, df):
        """Menambahkan fitur kustom lainnya"""
        # Triple MA Strategy signal
        df["ma_triple_bullish"] = (
            (df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])
        ).astype(int)
        df["ma_triple_bearish"] = (
            (df["ema_9"] < df["ema_21"]) & (df["ema_21"] < df["ema_50"])
        ).astype(int)

        # MACD Histogram changes
        if "MACDh_12_26_9" in df.columns:
            df["macd_hist_change"] = df["MACDh_12_26_9"].diff()
            df["macd_hist_acc"] = df["macd_hist_change"].diff()  # Acceleration
            df["macd_hist_sign_change"] = (
                (df["MACDh_12_26_9"] * df["MACDh_12_26_9"].shift(1)) < 0
            ).astype(int)

        # RSI features
        if "rsi_14" in df.columns:
            df["rsi_trend_up"] = (
                (df["rsi_14"] > 50) & (df["rsi_14"].shift(1) <= 50)
            ).astype(int)
            df["rsi_trend_down"] = (
                (df["rsi_14"] < 50) & (df["rsi_14"].shift(1) >= 50)
            ).astype(int)
            df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
            df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)

        # Bollinger Band features
        if all(col in df.columns for col in ["BBU_20_2.0", "BBL_20_2.0"]):
            df["bb_width"] = (df["BBU_20_2.0"] - df["BBL_20_2.0"]) / df["BBM_20_2.0"]
            df["bb_width_change"] = df["bb_width"].pct_change()

        # Combination features (interactions)
        if all(col in df.columns for col in ["trending_market", "rsi_14"]):
            df["trend_rsi_confirm"] = (
                (df["trending_market"] == 1)
                & (
                    (df["ma_triple_bullish"] == 1) & (df["rsi_14"] > 50)
                    | (df["ma_triple_bearish"] == 1) & (df["rsi_14"] < 50)
                )
            ).astype(int)

        # Trade entry signals based on multiple conditions
        if all(
            col in df.columns
            for col in ["rsi_14", "ma_cross_5_20", "macd_hist_sign_change"]
        ):
            df["buy_signal"] = (
                (df["rsi_14"] < 40)
                & (df["ma_cross_5_20"] == 1)
                & (df["macd_hist_sign_change"] == 1)
                & (df["macd_hist_change"] > 0)
            ).astype(int)

            df["sell_signal"] = (
                (df["rsi_14"] > 60)
                & (df["ma_cross_5_20"] == 0)
                & (df["macd_hist_sign_change"] == 1)
                & (df["macd_hist_change"] < 0)
            ).astype(int)

        return df

    def apply_pca(self, df, features):
        """Menerapkan PCA pada fitur-fitur yang dipilih"""
        if not self.use_pca:
            return df, features

        # Select only numeric columns for PCA
        numeric_df = df[features].select_dtypes(include=["float64", "int64"])

        if numeric_df.shape[1] <= self.pca_components:
            print(
                f"Warning: Number of features ({numeric_df.shape[1]}) is less than PCA components ({self.pca_components})"
            )
            return df, features

        # Scale the data
        scaled_data = self.scaler.fit_transform(numeric_df)

        # Apply PCA
        if self.pca is None:
            self.pca = PCA(n_components=self.pca_components)
            pca_result = self.pca.fit_transform(scaled_data)
        else:
            pca_result = self.pca.transform(scaled_data)

        # Create DataFrame with PCA results
        pca_columns = [f"pca_{i + 1}" for i in range(self.pca_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_columns, index=df.index)

        # Combine with original DataFrame
        result_df = pd.concat([df, pca_df], axis=1)

        # Update features list
        features = list(set(features) - set(numeric_df.columns)) + pca_columns

        return result_df, features

    def process(self, df, support_resistance=None, psychological_levels=None):
        """Proses utama untuk feature engineering"""
        # Buat salinan data
        df = df.copy()

        # Lakukan denoising jika diperlukan
        df = self.denoise_data(df)

        # Tambahkan fitur-fitur harga
        df = self.add_price_features(df)

        # Tambahkan indikator teknikal
        df = self.add_pandas_ta_indicators(df)

        # Tambahkan indikator dari TALib
        if self.use_talib:
            df = self.add_talib_indicators(df)

        # Tambahkan fitur support/resistance
        df = self.add_support_resistance_features(
            df, support_resistance, psychological_levels
        )

        # Tambahkan fitur regime pasar
        df = self.add_market_regime_features(df)

        # Tambahkan fitur waktu
        df = self.add_time_features(df)

        # Tambahkan fitur kustom
        df = self.add_custom_features(df)

        # Isi nilai NaN
        df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)

        # Pilih fitur yang akan digunakan (semua kolom kecuali target)
        features = [col for col in df.columns if col != "target"]

        # Terapkan PCA jika diperlukan
        if self.use_pca:
            df, features = self.apply_pca(df, features)

        return df, features
