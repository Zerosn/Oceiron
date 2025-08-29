import os
import time
import pandas as pd
import numpy as np
from random import randrange as rand
from data.colector import get_ticker_data, get_sector_tickers
from features import ticker_features as tf
from features import sector_features as sf
from model.model import init_model, load_checkpoint, load_ticker_data, train_model,predict_model
import torch
import datetime as dt
from tqdm import tqdm
import json
import logging

logging.basicConfig(
    level=logging.INFO,  # Nivel mínimo que se muestra
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),  # Guarda en archivo
        logging.StreamHandler()               # También en consola
    ]
)


def get_mode_config(history_mode=True, path="mode_config.json"):
    """
    Carga la configuración del modo de procesamiento desde un archivo JSON.
    - history_mode: Si es True, se usa el modo histórico (1 año), si
      no, modo corto plazo (3 meses)."""
    with open(path, "r") as f:
        configs = json.load(f)

    now = dt.datetime.now()

    if history_mode:
        cfg = configs["history_mode"]
    else:
        cfg = configs["short_mode"]

    # Resolver fechas dinámicas
    start = now - dt.timedelta(days=int(cfg["start_time"].replace("d_ago", "")))
    end = now if cfg["end_time"] == "now" else dt.datetime.fromisoformat(cfg["end_time"])

    cfg["start_time"] = start
    cfg["end_time"] = end
    cfg["ticker_list"] = "top_sectors.csv"
    return cfg

def model_config(path="model.json"):
    """
    Carga la configuración del modelo desde un archivo JSON.
    """
    with open(path, "r") as f:
        config = json.load(f)
    return config

def process_ticker(ticker, sector, start_time, end_time, use_max, folder, include_target=True):
    """
    Procesa un ticker específico, agregando características técnicas, estadísticas y de sector.
    - ticker: Símbolo del ticker a procesar.
    - sector: Sector al que pertenece el ticker.
    - start_time: Fecha de inicio para la recolección de datos.
    - end_time: Fecha de fin para la recolección de datos.
    - use_max: Si es True, se usa el máximo rango de datos disponible.
    - folder: Carpeta donde se guardarán los datos procesados."""
    df = get_ticker_data(
        ticker,
        start_time=start_time.strftime("%Y-%m-%d"),
        end_time=end_time.strftime("%Y-%m-%d"),
        use_max=use_max
    )

    if df is None or df.empty:
        logging.warning(f"Ticker {ticker} sin datos, se omite.")
        return
    if df.shape[0] < 20:
        logging.warning(f"Ticker {ticker} con menos de 20 datos, se omite.")
        return
    df = tf.add_technical_indicators(df)
    df = tf.add_statistical_features(df)
    df = sector_codification(df, sector)

    if include_target:
        df = create_target_class(df, horizon=3)

    df = tf.add_local_extrema_features(df)
    #df = tf.normalize_zscore(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert(None)
    df = tf.add_date_cyclic_features(df)

    df.to_csv(f"{folder}/{ticker}.csv", index=True)
    #print(f"Ticker {ticker} procesado y guardado.")

def process_all_tickers(history_mode=True, target_class=True):
    """
    Procesa todos los tickers listados en 'top100_sectors.csv', agregando características y guardándolos en archivos CSV.
    - history_mode: Si es True, se usa el modo histórico (1 año), si no, modo corto plazo (3 meses).
    - target_class: Si es True, se crea una columna objetivo basada en el retorno a 3 días.
    """
    config = get_mode_config(history_mode)
    ticker_list = config["ticker_list"]
    df_tickers = pd.read_csv(f"{ticker_list}", sep=";")
    os.makedirs(config["folder_tickers"], exist_ok=True)

    for sector in tqdm(df_tickers.columns, desc="processing tickers of sectors"):
        time.sleep(rand(3, 5))

        for ticker in tqdm(df_tickers[sector].dropna(),desc=f"Processing tickers in {sector}", leave=False):
            time.sleep(rand(1, 2))
            process_ticker(
                ticker,
                sector,
                config["start_time"],
                config["end_time"],
                config["use_max"],
                config["folder_tickers"],
                include_target=target_class
            )

def process_all_sectors(history_mode=True):
    """
    Procesa todas las características de los sectores basados en los tickers listados en 'top100_sectors.csv'.
    - history_mode: Si es True, se usa el modo histórico (1 año), si no, modo corto plazo (3 meses).
    """
    df_tickers = pd.read_csv("top100_sectors.csv", sep=";")
    sectors = df_tickers.columns
    config = get_mode_config(history_mode)
    folder = config["folder_tickers"]
    folder_sector = config["folder_sector"]
    os.makedirs(folder_sector, exist_ok=True)

    for sector in tqdm(sectors, desc="Create sectors feauteres"):
        sector_tickers = df_tickers[sector].dropna().tolist()
        sf.compute_sector_features(sector, sector_tickers, data_dir=folder, output_dir=folder_sector, cal_period=7)

def add_all_sector_features(history_mode=True):
    """
    Agrega características de sector a cada ticker basado en los datos procesados de los sectores.
    - history_mode: Si es True, se usa el modo histórico (1 año), si no, modo corto plazo (3 meses).
    """
    config = get_mode_config(history_mode)
    folder = config["folder_tickers"]
    folder_sector = config["folder_sector"]
    folder_output = config["folder_process_data"]
    os.makedirs(folder_output, exist_ok=True)
    df_tickers = pd.read_csv("top100_sectors.csv", sep=";")
    sectors = df_tickers.columns
    for sector in tqdm(sectors, desc="Adding sector features"):
        sector_tickers = df_tickers[sector].dropna().tolist()
        for ticker in sector_tickers:
            ticker_path = f"{folder}/{ticker}.csv"
            output_path = f"{folder_output}/{ticker}.csv"
            if os.path.exists(ticker_path):
                df_ticker = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
                df_ticker = sf.add_sector_features_to_ticker(df_ticker, sector, sector_dir=folder_sector)
                df_ticker.to_csv(output_path)

def tickers_clear(history_mode=True):
    """
    Limpia los datos de los tickers eliminando filas con valores NaN.
    - history_mode: Si es True, se usa el modo histórico (1 año), si no, modo corto plazo (3 meses).
    """
    df_tickers = pd.read_csv("top100_sectors.csv", sep=";")
    sectors = df_tickers.columns
    config = get_mode_config(history_mode)
    folder = config["folder_process_data"]

    for sector in tqdm(sectors,desc="Cleaning tickers data"):
        sector_tickers = df_tickers[sector].dropna().tolist()
        for ticker in tqdm(sector_tickers,desc=f"Cleaning tickers in {sector}", leave=False):
            try:
                ticker_path = f"{folder}/{ticker}.csv"
                if not os.path.exists(ticker_path):
                    logging.warning(f"Clear - Ticker {ticker} no existe, se omite.")
                    continue
                df_ticker = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
                df_ticker.dropna(inplace=True)
                df_ticker.to_csv(ticker_path)
            except Exception as e:
                logging.error(f"Clear - Ticker {ticker} error: {e}")

def normalizaton_zscore(history_mode = True,window=20):
    """
    Normaliza los datos de los tickers usando la normalización z-score.
    - history_mode: Si es True, se usa el modo histórico (1 año), si no, modo corto plazo (3 meses).
    - window: Ventana para el cálculo de la media y desviación estándar móvil.
    """
    sectors = df_tickers.columns
    config = get_mode_config(history_mode)
    ticker_list = config["ticker_list"]
    df_tickers = pd.read_csv(f"{ticker_list}", sep=";")
    folder = config["folder_process_data"]
    exclude = {'sector_cod', 'target',"Volume_Change","return_1d", 'return_5d', 'log_return_1d',
                'log_return_10d',"crossover_score","Price_above_ema","MACD_trend","bullish_score",
                "distance_from_sma","RSI_map","Gap","SMA_slope","breakout_up",
                "breakout_down","drawdown","Price_Acceleration","ATR","Returns_1", "Returns_5"}
    
    for sector in tqdm(sectors, desc="Normalization z-score", leave=False):
        #print(f"Normalization z-score data for: {sector}")
        sector_tickers = df_tickers[sector].dropna().tolist()
        for ticker in tqdm(sector_tickers,desc=f"Normalizing tickers in {sector}", leave=False):
            try:
                ticker_path = f"{folder}/{ticker}.csv"
                if not os.path.exists(ticker_path):
                    logging.warning(f"Normalizaton - Ticker {ticker} no existe, se omite.")
                    continue
                df_ticker = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
                numeric_cols = [col for col in df_ticker.select_dtypes(include="number").columns if col not in exclude]
                for col in tqdm(numeric_cols,desc="Normalizing columns",leave=False):
                    mean = df_ticker[col].rolling(window=window).mean()
                    std = df_ticker[col].rolling(window=window).std()
                    df_ticker[col] = (df_ticker[col] - mean) / std
                df_ticker.to_csv(ticker_path)
            except Exception as e:
                logging.error(f"Normalizing - Ticker {ticker} error: {e}")

def create_top_tickers_csv(count=100):
    """
    Crea un archivo CSV con los principales tickers de cada sector.
    - count: Número de tickers principales a incluir por sector.
    """
    sector_codes = ["sec-ind_sec-largest-equities_technology","sec-ind_sec-largest-equities_consumer-cyclical",
                    "sec-ind_sec-largest-equities_financial-services", "sec-ind_sec-largest-equities_communication-services",
                    "sec-ind_sec-largest-equities_healthcare", "sec-ind_sec-largest-equities_industrials",
                    "sec-ind_sec-largest-equities_energy", "sec-ind_sec-largest-equities_basic-materials",
                    "sec-ind_sec-largest-equities_real-estate", "sec-ind_sec-largest-equities_consumer-defensive",
                    "sec-ind_sec-largest-equities_utilities"]
    
    df = pd.DataFrame()
    for sector_code in sector_codes:
        sector = sector_code[29:]
        print(f"Processing sector: {sector}")
        tickers = get_sector_tickers(sector_code, count)
        if tickers:
            df[sector] = pd.Series(tickers)
        else:
            logging.warning(f"Create top - Sector {sector} sin Tickers, se omite.")
    config = get_mode_config(True)
    ticker_list = config["ticker_list"]
    df.to_csv(f"{ticker_list}", sep=';', index=False)

def sector_codification(df, sector):
    """
    Codifica el sector en una columna numérica entre -1 y 1.
    - df: DataFrame del ticker.
    - sector: Nombre del sector.
    """
    if "Sector" in df.columns:
        config = get_mode_config(True)
        ticker_list = config["ticker_list"]
        df_tickers = pd.read_csv(f"{ticker_list}", sep=";")
        sector_id = df_tickers.columns.get_loc(sector)
        n_sectors = len(df_tickers.columns)
        df["Sector_cod"] =  2*(sector_id / n_sectors) -1
        df = df.drop(columns=["Sector"])
    return df

def create_target_class(df, horizon=5):
    """
    Crea una columna objetivo basada en el retorno logarítmico a un horizonte dado.
    - df: DataFrame del ticker.
    - horizon: Horizonte en días para calcular el retorno.
    """
    df = df.copy()
    df["target"] = np.log(df["Close"].shift(-horizon) / df["Close"])
    return df

def data_pipeline(history_mode=True,target_class=True):
    """
    Ejecuta el pipeline completo de procesamiento de datos.
    - history_mode: Si es True, se usa el modo histórico (1 año), si no, modo corto plazo (3 meses).
    """
    steps = [
        ("Procesando tickers", process_all_tickers),
        ("Procesando sectores", process_all_sectors),
        ("Agregando features de sectores", add_all_sector_features),
        ("Normalizando datos", normalizaton_zscore),
        ("Limpiando tickers", tickers_clear),
        
    ]

    for desc, func in tqdm(steps, desc="Ejecutando pipeline", unit="etapa"):
        if func == process_all_tickers:
            func(history_mode=history_mode, target_class=target_class)
        elif func == normalizaton_zscore:
            func(history_mode, window=20)
        else:
            func(history_mode)

def check_tickers(history_mode=True):
    """
    Verifica la integridad de los archivos de tickers procesados.
    - history_mode: Si es True, se usa el modo histórico (1 año), si no, modo corto plazo (3 meses).
    Crea un archivo 'check_tickers.csv' con el estado de cada ticker.
    """
    sectors = df_tickers.columns
    config = get_mode_config(history_mode)
    folder = config["folder_process_data"]
    ticker_list = config["ticker_list"]
    df_tickers = pd.read_csv(f"{ticker_list}", sep=";")
    df_check = pd.DataFrame()
    check_tickers =[]
    for sector in tqdm(sectors, desc="Checking tickers"):
        sector_tickers = df_tickers[sector].dropna().tolist()
        for ticker in tqdm(sector_tickers, desc=f"Checking tickers in {sector}", leave=False):
            try:
                df = pd.read_csv(f"{folder}/{ticker}.csv", index_col=0, parse_dates=True)
                if df.empty:
                    logging.warning(f"check tickers - Ticker {ticker} Está vacío.")
                check_tickers.append([ticker,len(df.columns)])
            except FileNotFoundError:
                logging.warning(f"check tickers - Ticker {ticker} file not found.")
                
        
    df_check = pd.DataFrame(check_tickers, columns=["Ticker", "Columns"])
    df_check.to_csv(f"check_tickers.csv", index=False)

def csv_to_parquet(history_mode=True):
    """
    Convierte los archivos CSV procesados a formato Parquet.
    - history_mode: Si es True, se usa el modo histórico (1 año), si no, modo corto plazo (3 meses).
    """
    config = get_mode_config(history_mode)
    folder = config["folder_process_data"]
    output_folder = folder + "_parquet"
    os.makedirs(output_folder, exist_ok=True)
    for file in tqdm(os.listdir(folder), desc="Converting CSV to Parquet"):
        if file.endswith(".csv"):
            csv_path = os.path.join(folder, file)
            parquet_path = os.path.join(output_folder, file.replace(".csv", ".parquet"))
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                df.to_parquet(parquet_path)
            except Exception as e:
                logging.error(f"Error al convertir {file}: {e}")

def create_model(model_path, history_mode=True):
    """
    Crea y devuelve un modelo RNN con atención, cargando un checkpoint si existe.
    - checkpoint_path: Ruta al archivo de checkpoint.
    - history_mode: Si es True, se usa el modo histórico (1 año), si no, modo corto plazo (3 meses).
    Retorna el modelo, optimizador, función de pérdida y scheduler.
    """
    config = model_config()
    model, optimizer, loss_fn, scheduler = init_model(
        config["input_size"],
        config["hidden_size"],
        config["num_layers"],
        config["dropout"],
        config["learning_rate"]
    )
    
    start_epoch, ticker_start, history = load_checkpoint(model_path, model, optimizer)
    
    return model, optimizer, loss_fn, scheduler, start_epoch, ticker_start, history

def train_model_pipeline(model, optimizer, loss_fn, scheduler,start_epoch,
                        ticker_start, data_folder,checkpoint_path,parquet=True):
    """
    Entrena el modelo RNN con atención usando los datos de tickers procesados.
    - model: Modelo RNN con atención.
    - optimizer: Optimizador para el entrenamiento.
    - loss_fn: Función de pérdida.
    - scheduler: Scheduler para ajustar la tasa de aprendizaje.
    - start_epoch: Época de inicio para el entrenamiento.
    - ticker_start: Ticker donde retoma el entrenamiento.
    - history: Historial de pérdidas de entrenamiento y validación.
    - data_folder: Carpeta donde se encuentran los datos procesados de los tickers.
    - checkpoint_path: Ruta al archivo de checkpoint para guardar el estado del modelo.
    """
    config = model_config()
    df_tickers = pd.read_csv(f'{data_folder}/ticker_partial_shuffle.csv')
    ticker_num = len(df_tickers)

    for ticker_id in range(ticker_start, ticker_num):
        ticker = df_tickers["Ticker"][ticker_id]
        start_index = df_tickers["start_index"][ticker_id]
        end_index = df_tickers["end_index"][ticker_id]
        if parquet:
            ticker_path = f"{data_folder}/process_data_parquet/{ticker}.parquet"
        else:
            ticker_path = f"{data_folder}/process_data/{ticker}.csv"

        if not os.path.exists(ticker_path):
            logging.warning(f"train_model_pipeline - Ticker {ticker} no encontrado, saltando.")
            continue

        X_tensor, y_tensor = load_ticker_data(ticker_path, start_index, end_index, config['window_size'])
        history = train_model(
            model, optimizer, loss_fn, scheduler,
            X_tensor, y_tensor,
            config['total_epochs'], config['batch_size'], start_epoch,
            checkpoint_path, history, ticker_id
        )

def model_prediction(model, ticker_path, model_config, start_index=None, end_index=None):
    """
    Realiza predicciones con un modelo RNN con atención ya cargado.
    """

    if end_index is None:
        df_ticker = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
        end_index = len(df_ticker)
    if start_index is None:
        start_index = max(0, end_index - model_config['window_size'])

    X_tensor, _ = load_ticker_data(ticker_path, start_index, end_index, model_config['window_size'], target_col=False)
    y_pred = predict_model(model, X_tensor, model_config['batch_size'])
    return y_pred

def create_batches(ticker_tensors, batch_size):
    """
    Crea batches de tensores a partir de una lista de (ticker, tensor).
    Retorna una lista de batches, donde cada batch es:
        (tickers, X_batch)
    """
    batches = []
    for i in range(0, len(ticker_tensors), batch_size):
        batch = ticker_tensors[i:i+batch_size]
        tickers = [t[0] for t in batch]
        tensors = [t[1] for t in batch]

        # concatenamos sobre el eje 0 (batch)

        X_batch = torch.cat(tensors, dim=0)  # (B, seq_len, features)
        batches.append((tickers, X_batch))
    return batches

def predict_model_pipeline(model_path, start_index=None, end_index=None):
    """
    Realiza predicciones en batches para todos los tickers procesados.
    """
    config = model_config()
    mode = get_mode_config(history_mode=False)
    ticker_list = config["ticker_list"]
    df_tickers = pd.read_csv(f"{ticker_list}", sep=";")

    # 🔹 Cargar el modelo UNA SOLA VEZ
    model, optimizer, loss_fn, scheduler, start_epoch, ticker_start, history = create_model(model_path)
    model.eval()

    # Primero preparamos todos los tensores
    ticker_tensors = []
    for sector in tqdm(df_tickers.columns, desc="loading tickers of sectors"):
        for ticker in tqdm(df_tickers[sector].dropna(), desc=f"Loading tickers in {sector}", leave=False):
            ticker_path = f"{mode['folder_process_data']}/{ticker}.csv"

            if not os.path.exists(ticker_path):
                logging.warning(f"Ticker {ticker} no encontrado, saltando. ({ticker_path})")
                continue

            # determinamos índices
            df_ticker = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
            if end_index is None:
                local_end = len(df_ticker)
            else:
                local_end = end_index
            if start_index is None:
                local_start = max(0, local_end - config['window_size'])
            else:
                local_start = start_index

            # cargar tensor
            X_tensor, _ = load_ticker_data(ticker_path, local_start, local_end, config['window_size'], target_col=False)
            if X_tensor is None:
                break
            ticker_tensors.append((ticker, X_tensor))

    # 🔹 Crear batches

    batches = create_batches(ticker_tensors, config['batch_size'])

    # 🔹 Ejecutar modelo en batches
    Predictions = []
    for tickers, X_batch in tqdm(batches, desc="predicting batches"):
        with torch.no_grad():

            y_pred = predict_model(model, X_batch, batch_size=len(tickers))
        
        # y_pred puede ser (B, output_dim), lo separamos ticker a ticker
        for t, pred in zip(tickers, y_pred):
            Predictions.append((t, pred.numpy().tolist()))

    return Predictions

def target_to_percentage(y):
    """
    Convierte un valor de retorno logarítmico a porcentaje.
    """
    return (np.exp(y) - 1) * 100

def create_ticker_partial_shuffle(history_mode=False,partial_length=100,overlap=20):
    """
    Crea un archivo CSV con tickers y sus índices de inicio y fin para entrenamiento.
    - history_mode: Si es True, se usa el modo histórico (toda la informacion disponible), si no, modo corto plazo (3 meses).
    """
    seed=31415
    config = get_mode_config(history_mode)
    folder = config["folder_process_data"]
    ticker_list = config["ticker_list"]
    df_tickers = pd.read_csv(f"{ticker_list}", sep=";")
    sectors = df_tickers.columns
    df_partial = pd.DataFrame(columns=["Ticker", "start_index", "end_index"])
    results = []
    for sector in tqdm(sectors, desc="Creating ticker partial shuffle"):
        sector_tickers = df_tickers[sector].dropna().tolist()
        for ticker in tqdm(sector_tickers, desc=f"Creating tickers in {sector}", leave=False):
            try:
                ticker_path = f"{folder}/{ticker}.csv"
                if not os.path.exists(ticker_path):
                    logging.warning(f"Partial - Ticker {ticker} no existe, se omite.")
                    continue
                df_ticker = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
                n_rows = len(df_ticker)
                start_index = 0
                for i in range(0,n_rows-overlap,partial_length):
                    end_index = min(i+partial_length, n_rows)
                    results.append({
                        "Ticker": ticker,
                        "start_index": start_index,
                        "end_index": end_index
                    })
                    start_index = end_index - overlap
            except Exception as e:
                logging.error(f"Partial - Ticker {ticker} error: {e}")
    df_partial = pd.DataFrame(results)
    df_partial = df_partial.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_partial.to_csv(f"{folder}/ticker_partial_shuffle.csv", index=False)
    logging.info(f"rchivo ticker_partial_shuffle.csv creado con {len(df_partial)} entradas.")
    logging.info(f"archivo guardado en: {folder}/ticker_partial_shuffle.csv")

def clear_log_file(log_file="pipeline.log"):
    """
    Limpia el archivo de log.
    """
    with open(log_file, "w") as f:
        f.write("")

def clear_data_folders(history_mode=True):
    """
    Limpia las carpetas de datos procesados.
    - history_mode: Si es True, se usa el modo histórico (1 año), si no, modo corto plazo (3 meses).
    """
    config = get_mode_config(history_mode)
    folders = [config["folder_tickers"], config["folder_sector"], config["folder_process_data"], config["folder_process_data"]+"_parquet"]
    for folder in folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logging.error(f"Error al eliminar {file_path}: {e}")
            logging.info(f"Carpeta {folder} limpiada.")
        else:
            logging.info(f"Carpeta {folder} no existe, se omite.")

if __name__ == "__main__":
    create_ticker_partial_shuffle()