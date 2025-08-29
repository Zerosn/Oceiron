import yfinance as yf
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup


def get_ticker_data(ticker_name,start_time="2010-01-01", end_time="2023-10-01",use_max=True):
    try:
        ticker = yf.Ticker(ticker_name)
        if use_max:
            df = ticker.history(period="max")
        else:
            df = ticker.history(start=start_time, end=end_time)
        df.drop(columns=["Dividends", "Stock Splits"],errors="ignore", inplace=True)
        return df
    except Exception as e:
        print(f"Error downloading {ticker_name}: {e}")
        return None

def get_ticker_name(sector_id=0, ticker_id=0,data_dir="data"):
#r"D:/Documentos D/GitHub/Oceiron/top100_sectors.csv"
    df = pd.read_csv(data_dir, sep=";")
    sector_columns = df.columns.to_list()
    return df.iloc[ticker_id][sector_columns[sector_id]]

def get_sector_tickers(sector_code="sec-ind_sec-largest-equities_technology", count=100):
    """
    Obtiene los tickers de un sector desde Yahoo Finance usando BeautifulSoup.
    """
    url = f"https://finance.yahoo.com/research-hub/screener/{sector_code}/?count={count}&guccounter=1"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    try:
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"Error HTTP {r.status_code}")
            return []

        soup = BeautifulSoup(r.text, 'html.parser')

        tickers = []
        for tag in soup.find_all('span', class_=re.compile(r'^symbol yf-')):
            if not tag.find_parent('div', class_=re.compile(r'^container-wrapper yf')):
                ticker = tag.get_text(strip=True)
                if ticker:
                    tickers.append(ticker)
                if len(tickers) >= count:
                    break

        if not tickers:
            print("No se encontraron tickers. Puede que la página haya cambiado su estructura.")
        return tickers # Limitar a 'count' tickers

    except Exception as e:
        print(f"Error obteniendo tickers para {sector_code}: {e}")
        return []
