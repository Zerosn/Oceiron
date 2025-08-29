import pandas as pd
import os

def compute_sector_features(
    sector_id,
    sector_tickers,
    data_dir="data",
    output_dir="sector_data",
    cal_period=5,
    norm_window=20
):
    ticker_returns = {}
    for ticker in sector_tickers:
        path = os.path.join(data_dir, f"{ticker}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.sort_index()
            #df.index = pd.to_datetime(df.index, errors='coerce')  # Convierte a datetime, incluso si es string
            #df.index = df.index.tz_localize(None, ambiguous='NaT', nonexistent='NaT')  # Si es tz-aware

            returns = df["Close"].pct_change().dropna()
            # Almacenar retornos
            ticker_returns[ticker] = returns

    # Combinar returns
    sector_returns = pd.DataFrame(ticker_returns)

    # Mantener solo fechas donde hay al menos un valor
    sector_returns = sector_returns.dropna(how="all")

    # Cálculo de promedio y volatilidad
    sector_returns["sector_avg_return"] = sector_returns.mean(axis=1, skipna=True)
    sector_returns[f"sector_avg_return_{cal_period}d"] = sector_returns["sector_avg_return"].rolling(window=cal_period).mean()
    sector_returns[f"sector_volatility_{cal_period}d"] = sector_returns["sector_avg_return"].rolling(window=cal_period).std()

    # Limpiar fechas donde rolling genera NaN
    df_sector = sector_returns[[f"sector_avg_return_{cal_period}d", f"sector_volatility_{cal_period}d"]].dropna()

    # Normalización z-score
    normalize_zscore(df_sector, window=cal_period)

    # Último filtrado y orden
    df_sector = df_sector.dropna()
    df_sector = df_sector.sort_index()

    # Guardar
    os.makedirs(output_dir, exist_ok=True)
    df_sector.to_csv(os.path.join(output_dir, f"{sector_id}_features.csv"))

    return df_sector
    
    
    normalize_zscore(df_sector, window=20) #mismo valo que en ticker_features.py

    df_sector.dropna(inplace=True)  # Solo si no hay forma de reindexar
    os.makedirs(output_dir, exist_ok=True)
    df_sector.to_csv(f"{output_dir}/Sector_{sector_id}.csv")

def normalize_zscore(df, window=20):
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        mean = df[col].rolling(window=window).mean()
        std = df[col].rolling(window=window).std()
        df[col] = (df[col] - mean) / std
    return df

def add_sector_features_to_ticker(ticker_df, sector_name, sector_dir="sector_data"):
    df_sector = pd.read_csv(f"{sector_dir}/{sector_name}_features.csv", index_col=0, parse_dates=True)
    overlapping_cols = df_sector.columns.intersection(ticker_df.columns)
    if len(overlapping_cols) > 0:
        ticker_df = ticker_df.drop(columns=overlapping_cols)

    df_combined = ticker_df.join(df_sector, how="left")
    return df_combined


