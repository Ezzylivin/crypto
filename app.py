import os
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from coinbase.rest import RESTClient  # 🚀 Using Official Library
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Allow Vercel frontend to access this API
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Configuration ---
SUPPORTED_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", 
    "DOGE-USD", "AVAX-USD", "LINK-USD", "MATIC-USD", "LTC-USD", 
    "DOT-USD", "SHIB-USD", "UNI-USD"
]

TOP_5_SYMBOLS = ["BTC-USD", "ETH-USD"]

FRED_SERIES_IDS = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL"
}

# --- Helper to Fix NaN JSON Crash ---
def clean_nan(obj):
    """Recursively replace NaN/Infinity with None (null in JSON)."""
    if isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj

# --- Data Fetching ---
def get_fred_data(start_date, end_date, api_key):
    """Fetches macro data from FRED."""
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
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json().get('observations', [])
                df = pd.DataFrame(data)[['date', 'value']]
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                all_series_data[name] = df['value']
        except:
            all_series_data[name] = pd.Series(dtype=float)
    
    macro_df = pd.DataFrame(all_series_data)
    macro_df.ffill(inplace=True)
    return macro_df

def get_single_symbol_data(symbol, start_time, end_time, granularity="ONE_DAY"):
    """Fetches OHLCV data using coinbase-advanced-py."""
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    
    if not api_key or not api_secret:
        print("Error: Missing Coinbase Credentials")
        return None

    try:
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        
        # Convert timestamps to string integers (required by this lib)
        start_ts = str(int(start_time.timestamp()))
        end_ts = str(int(end_time.timestamp()))
        
        # Fetch Candles
        response = client.get_candles(
            product_id=symbol,
            start=start_ts,
            end=end_ts,
            granularity=granularity
        )
        
        # Parse the 'candles' list from the response object
        if not hasattr(response, 'candles') or not response.candles:
            return None

        # Convert candle objects to dicts
        candles_data = []
        for c in response.candles:
            candles_data.append({
                "start": float(c.start),
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume)
            })

        df = pd.DataFrame(candles_data)
        # Ensure we have data before processing
        if df.empty: return None
        
        df['time'] = pd.to_datetime(df['start'], unit='s')
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        return df

    except Exception as e:
        print(f"[COINBASE ERROR] {symbol}: {e}")
        return None

def get_top_market_data():
    """Fetches combined data for initial dashboard load."""
    fred_api_key = os.environ.get("FRED_API_KEY")
    end_time = datetime.now(timezone.utc)
    # Fetch 90 days for daily charts is fine (90 < 350)
    start_time = end_time - timedelta(days=90)
    
    all_crypto_data = {}
    macro_df = get_fred_data(start_time, end_time, fred_api_key) if fred_api_key else pd.DataFrame()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(get_single_symbol_data, sym, start_time, end_time, "ONE_DAY"): sym for sym in TOP_5_SYMBOLS}
        for future in futures:
            sym = futures[future]
            try:
                df = future.result()
                if df is not None:
                    if not macro_df.empty:
                        df = df.join(macro_df, how='left').ffill()
                    
                    # 🚀 FIX: Convert NaN to None immediately here
                    df = df.replace({np.nan: None})
                    all_crypto_data[sym] = df
            except: pass
    return all_crypto_data

# --- API ENDPOINTS ---

@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    return jsonify(SUPPORTED_SYMBOLS)

@app.route('/api/candles', methods=['GET'])
def get_candles():
    symbol = request.args.get('product_id', 'BTC-USD')
    # Frontend sends 'ONE_HOUR', 'ONE_DAY' etc.
    granularity = request.args.get('granularity', 'ONE_HOUR')
    
    end_time = datetime.now(timezone.utc)
    
    # 🚀 FIX: RESPECT COINBASE 350 CANDLE LIMIT
    # Hourly: 350 hours ~= 14.5 days. Use 14 days max.
    # Daily: 350 days max. Use 300 days.
    if granularity == 'ONE_HOUR':
        days_back = 14 
    else:
        days_back = 300 
        
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
    
    # Clean NaNs before sending JSON
    return jsonify(clean_nan(records))

@app.route('/api/data', methods=['GET'])
def get_market_data_api():
    """Legacy Endpoint: Returns heavy payload for initial load."""
    try:
        data_dict = get_top_market_data()
        json_payload = {}
        for symbol, df in data_dict.items():
            df_reset = df.reset_index()
            records = df_reset.to_dict(orient='records')
            
            cleaned_records = []
            for r in records:
                r['time'] = r['time'].isoformat() if isinstance(r['time'], pd.Timestamp) else str(r['time'])
                cleaned_records.append(r)
                
            json_payload[symbol] = clean_nan(cleaned_records)
            
        return jsonify(json_payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
