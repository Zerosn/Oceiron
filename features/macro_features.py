import yfinance as yf
import pandas as pd
import os
def get_macro_features(
    start_time="2000-08-23",
    end_time="2025-10-01",
    use_max=True,
    data_dir="data/macro_data"
):
    """
    Descarga y calcula características macroeconómicas clave.
    
    Args:
        start_time (str): Fecha de inicio para los datos.
        end_time (str): Fecha de fin para los datos.
        use_max (bool): Si True, usa el periodo máximo disponible.
        data_dir (str): Directorio donde se guardarán los datos descargados.
    
    Returns:
        pd.DataFrame: DataFrame con las características macroeconómicas.
    """
    # Lista de activos macroeconómicos clave
    tickers = [
        "^TNX",     # 10 Year Treasury Yield
        "^IRX",     # 13 Week Treasury Bill Rate (proxy para tasa corta)
        "DX=F",     # US Dollar Index Futures
        "GC=F",     # Gold Futures
        "CL=F",     # Crude Oil Futures
        "HG=F",     # Copper Futures
        "^VIX",     # Volatility Index (Fear Index)
        "^GSPC",    # S&P 500 Index
        "^NDX",     # Nasdaq 100
        "^RUT",     # Russell 2000
        "TLT"       # Long-Term Treasury ETF (proxy para bonos largos)
    ]

    # Descargar datos ajustados de cierre
    if use_max:
        macro_data = yf.download(tickers,period="max")["Close"]
    else:
        macro_data = yf.download(tickers, start=start_time, end=end_time)["Close"]

    # Guardar los datos en el directorio especificado
    os.makedirs(data_dir, exist_ok=True)
    macro_data.to_csv(f"{data_dir}/macro_features.csv")

    return macro_data

def process_macro_features(df_macro,window=5):
    """
    Procesa las características macroeconómicas para calcular retornos y normalizar.
    
    Args:
        df_macro (pd.DataFrame): DataFrame con los datos de características macroeconómicas.
    
    Returns:
        pd.DataFrame: DataFrame con las características procesadas.
    """
    z_Score_features = ["^TNX", "DX=F", "GC=F", "CL=F","BZ=F", "HG=F", "^IRX", "^VIX", "TLT"]
    df_return =pd.DataFrame(index=df_macro.index)
    for feature in z_Score_features:
        df_return[f"{feature}_zscore"] = (df_macro[feature] - df_macro[feature].rolling(4*window).mean()) / df_macro[feature].rolling(4*window).std()
    features_pct = ["DX=F", "GC=F", "CL=F", "HG=F","^VIX", "^GSPC", "^NDX", "^RUT","BZ=F"]
    for feature in features_pct:
        df_return[f"{feature}_pct"] = df_macro[feature].pct_change(periods=window)
    df_return["VIX_momentum"] = df_macro["^VIX"].diff()
    df_return["Gold_momentum"] = df_macro["GC=F"].diff()
    df_return["Rate_momentum"] = df_macro["^TNX"].diff()
    df_return["yield_spread"] = df_macro["^TNX"] - df_macro["^IRX"]
    df_return.dropna(inplace=True)
    return df_return
    
