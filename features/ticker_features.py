import numpy as np
import pandas as pd
from ta import volume, momentum, volatility, trend
import yfinance as yf
from scipy.signal import argrelextrema

def add_technical_indicators(df):
    standar_window=5
    original_len = len(df)
    df["RSI"] = momentum.RSIIndicator(df["Close"], window=2*standar_window).rsi()
    df["ROC"] = momentum.ROCIndicator(df["Close"], window=3*standar_window).roc()
    df["MACD"] = trend.MACD(df["Close"]).macd()
    df["EMA"] = trend.EMAIndicator(df["Close"], window=3*standar_window).ema_indicator()
    df["SMA_1sw"] = trend.SMAIndicator(df["Close"], window=standar_window).sma_indicator()
    df["SMA_2sw"] = trend.SMAIndicator(df["Close"], window=2*standar_window).sma_indicator()
    df["SMA_3sw"] = trend.SMAIndicator(df["Close"], window=3*standar_window).sma_indicator()
    bb = volatility.BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["DC_High"] = volatility.donchian_channel_hband(df["High"], df["Low"], df["Close"], window=3*standar_window)
    df["DC_Low"] = volatility.donchian_channel_lband(df["High"], df["Low"], df["Close"], window=3*standar_window)
    df['DC_Mid'] = volatility.donchian_channel_mband(df["High"], df["Low"], df["Close"], window=3*standar_window)
    df["bollinger_bandwidth"] = df["BB_High"] - df["BB_Low"]
    df["breakout_up"] = (df["Close"] > df["DC_High"]).astype(int)
    df["breakout_down"] = (df["Close"] < df["DC_Low"]).astype(int)
    df["rolling_max"] = df["Close"].rolling(window=10).max()
    df["drawdown"] = (df["Close"] - df["rolling_max"]) / df["rolling_max"]
    df['ATR'] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df["FI"] = volume.ForceIndexIndicator(df["Close"], df["Volume"]).force_index()
    df["Price_Acceleration"] = df["Close"].pct_change().diff()
    df["SMA_slope"] = df["SMA_3sw"].pct_change()
    df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df[["Price_Acceleration", "SMA_slope", "Gap"]] = df[["Price_Acceleration", "SMA_slope", "Gap"]].fillna(0)
    df["Volume_Change"] = df["Volume"].pct_change().fillna(0)
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)
    df['log_return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_return_10d'] = np.log(df['Close'] / df['Close'].shift(10))
    df['crossover_5_15'] = (df['SMA_1sw'] > df['SMA_2sw']).astype(int)
    df['crossover_5_20'] = (df['SMA_1sw'] > df['SMA_3sw']).astype(int)
    df['crossover_score'] = df['crossover_5_15'] + df['crossover_5_20']
    df['Price_above_ema'] = (df['Close'] > df['EMA']).astype(int)
    df['MACD_trend'] = (df['MACD'] > 0).astype(int)
    df["bullish_score"] = ((df['RSI'] > 50).astype(int) +
                           (df['MACD'] > 0).astype(int) +
                           (df['Close'] > df['EMA']).astype(int)
                           )
    df["distance_from_sma"] = (df["Close"] - df["SMA_1sw"]) / df["SMA_1sw"]
    df["RSI_map"] = df["RSI"].apply(lambda x: (x / 50 - 1) if pd.notna(x) else 0)
    df = df.drop(columns=["RSI","crossover_5_15", "crossover_5_20",], errors='ignore')
    return df

def add_statistical_features(df, std_periods=[5,10,15], return_periods=[1,5]):
    for p in std_periods:
        df[f"std_{p}"] = df["Close"].rolling(window=p).std()
        df[f"Volatility_{p}"] = df["Close"].rolling(window=p).std()
    for p in return_periods:
        df[f"Returns_{p}"] = df["Close"].pct_change(periods=p, fill_method=None)
    return df

def normalize_zscore(df, window=20):
    exclude = {'sector_cod', 'target',"Volume_Change","return_1d", 'return_5d', 'log_return_1d',
                'log_return_10d',"crossover_score","Price_above_ema","MACD_trend","bullish_score",
                "distance_from_sma","RSI_map","Gap","SMA_slope","breakout_up",
                "breakout_down","drawdown","Price_Acceleration","ATR","Returns_1", "Returns_5"}
    numeric_cols = [col for col in df.select_dtypes(include="number").columns if col not in exclude]

    for col in numeric_cols:
        mean = df[col].rolling(window=window).mean()
        std = df[col].rolling(window=window).std()
        df[col] = (df[col] - mean) / std
    return df

def add_date_cyclic_features(df):
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["weekday"] = df.index.weekday
    
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df.drop(columns=["day", "month", "weekday"], inplace=True)
    return df

def add_local_extrema_features(df, column='Close', order=5):
    """
    Agrega columnas que identifican máximos y mínimos locales y su distancia/retorno desde ellos.
    """
    df = df.copy()

    # Inicializar columnas
    df['local_max'] = 0
    df['local_min'] = 0

    # Detectar máximos y mínimos locales
    local_max_idx = argrelextrema(df[column].values, np.greater_equal, order=order)[0]
    local_min_idx = argrelextrema(df[column].values, np.less_equal, order=order)[0]

    df.loc[df.index[local_max_idx], 'local_max'] = 1
    df.loc[df.index[local_min_idx], 'local_min'] = 1

    # Días desde último máximo o mínimo local
    df['days_since_local_max'] = (df['local_max'].cumsum() != 0).astype(int).cumsum() - df['local_max'].cumsum()
    df['days_since_local_min'] = (df['local_min'].cumsum() != 0).astype(int).cumsum() - df['local_min'].cumsum()

    # Retorno desde último máximo o mínimo local
    df['return_from_local_max'] = df[column] / df[column].where(df['local_max'] == 1).ffill() - 1
    df['return_from_local_min'] = df[column] / df[column].where(df['local_min'] == 1).ffill() - 1
    return df