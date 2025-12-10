import os
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from coinbase.rest import RESTClient
from concurrent.futures import ThreadPoolExecutor
import re

app = Flask(__name__)

# Allow Vercel frontend to access this API
vercel_regex = r"^https:\/\/.*\.vercel\.app$"
CORS(app, resources={r"/api/*": {"origins": vercel_regex, "supports_credentials": True}})

# --- 1. CONFIGURATION ---

# 🚀 NEW: The list for your Dashboard Dropdown
SUPPORTED_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", 
    "DOGE-USD", "AVAX-USD", "LINK-USD", "MATIC-USD", "LTC-USD", 
    "DOT-USD", "SHIB-USD", "UNI-USD", "ATOM-USD", "XLM-USD"
]

# Keep Top 2 fast for the initial landing load
TOP_5_SYMBOLS = ["BTC-USD", "ETH-USD"]

FRED_SERIES_IDS = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL"
}

# --- Data Fetching Functions ---
def get_fred_data(start_date, end_date, api_key):
    """Fetches and processes real macroeconomic data from the FRED API."""
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    today = datetime.now(timezone.utc).date()
    if end_date.date() > today:
        end_date = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
    
    all_series_data = {}
    for name, series_id in FRED_SERIES_IDS.items():
        try:
            params = {
                "series_id": series_id, "api_key": api_key, "file_type": "json",
                "observation_start": start_date.strftime('%Y-%m-%d'),
                "observation_end": end_date.strftime('%Y-%m-%d'),
            }
            response = requests.get(base_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json().get('observations', [])
                df = pd.DataFrame(data)[['date', 'value']]
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                all_series_data[name] = df['value']
        except Exception as e:
            print(f"[FRED ERROR] {name}: {e}")
            all_series_data[name] = pd.Series(dtype=float)
    
    macro_df = pd.DataFrame(all_series_data)
    macro_df.ffill(inplace=True)
    return macro_df

def get_single_symbol_data(symbol, start_time, end_time, granularity="ONE_DAY"):
    """Fetches historical price data for one symbol from Coinbase."""
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    if not api_key or not api_secret: return None

    client = RESTClient(api_key=api_key, api_secret=api_secret)
    try:
        response = client.get_candles(
            product_id=symbol,
            start=str(int(start_time.timestamp())),
            end=str(int(end_time.timestamp())),
            granularity=granularity
        )
        candle_dicts = [{"start": c.start, "high": c.high, "low": c.low, "open": c.open, "close": c.close, "volume": c.volume} for c in response.candles]
        if not candle_dicts: return None
        
        df = pd.DataFrame(candle_dicts)
        # Convert start (seconds) to datetime
        df['time'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s')
        df.set_index('time', inplace=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"[COINBASE ERROR] {symbol}: {e}")
        return None

def get_top_market_data():
    """Fetches combined data for the main dashboard load."""
    fred_api_key = os.environ.get("FRED_API_KEY")
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)
    all_crypto_data = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        fred_future = executor.submit(get_fred_data, start_time, end_time, fred_api_key) if fred_api_key else None
        coinbase_futures = {executor.submit(get_single_symbol_data, sym, start_time, end_time): sym for sym in TOP_5_SYMBOLS}
        
        macro_df = fred_future.result() if fred_future else pd.DataFrame()
        
        for future in coinbase_futures:
            symbol = coinbase_futures[future]
            try:
                symbol_df = future.result()
                if symbol_df is not None:
                    combined = symbol_df.join(macro_df, how='left').ffill() if not macro_df.empty else symbol_df
                    all_crypto_data[symbol] = combined
            except: pass
    return all_crypto_data

# --- API ENDPOINTS ---

# 🚀 NEW: Endpoint for the Dashboard Dropdown
@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    return jsonify(SUPPORTED_SYMBOLS)

# 🚀 NEW: Endpoint for the Selected Chart (Hourly Data)
@app.route('/api/candles', methods=['GET'])
def get_candles():
    symbol = request.args.get('product_id', 'BTC-USD')
    # Default to hourly for better charts, or daily if requested
    granularity = request.args.get('granularity', 'ONE_HOUR')
    
    end_time = datetime.now(timezone.utc)
    # Fetch 30 days of hourly data or 365 days of daily data
    days_back = 30 if granularity == 'ONE_HOUR' else 365
    start_time = end_time - timedelta(days=days_back)

    df = get_single_symbol_data(symbol, start_time, end_time, granularity)
    
    if df is None or df.empty:
        return jsonify([])
    
    # Format for Frontend
    df_reset = df.reset_index()
    records = []
    for _, row in df_reset.iterrows():
        records.append({
            "start": int(row['start']), 
            "time": row['time'].isoformat(),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume']
        })
    
    return jsonify(records)

@app.route('/api/data', methods=['GET'])
def get_market_data_api():
    """Legacy Endpoint: Returns heavy payload for initial load (Metric Cards + BTC/ETH)."""
    try:
        data_dict = get_top_market_data()
        json_payload = {}
        for symbol, df in data_dict.items():
            df_reset = df.reset_index()
            json_payload[symbol] = []
            for _, row in df_reset.iterrows():
                entry = row.to_dict()
                entry['time'] = row['time'].isoformat()
                if 'start' in entry: entry['start'] = int(entry['start'])
                json_payload[symbol].append(entry)
        return jsonify(json_payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
