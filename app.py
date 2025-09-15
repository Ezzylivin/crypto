import os
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from coinbase.rest import RESTClient
from concurrent.futures import ThreadPoolExecutor
import re # 🛠️ Import the regular expression library

app = Flask(__name__)

# 🛠️ Updated CORS configuration to accept any .vercel.app subdomain
# This uses a regular expression to match origins dynamically.
vercel_regex = r"^https:\/\/.*\.vercel\.app$"
CORS(app, resources={r"/api/*": {"origins": vercel_regex, "supports_credentials": True}})

# --- Configuration ---
TOP_5_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]
FRED_SERIES_IDS = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL"
}

# --- Data Fetching Functions ---

def get_fred_data(start_date, end_date, api_key):
    """Fetches and processes real macroeconomic data from the FRED API."""
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    all_series_data = {}
    for name, series_id in FRED_SERIES_IDS.items():
        params = {
            "series_id": series_id, "api_key": api_key, "file_type": "json",
            "observation_start": start_date.strftime('%Y-%m-%d'),
            "observation_end": end_date.strftime('%Y-%m-%d'),
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()['observations']
        df = pd.DataFrame(data)[['date', 'value']]
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        all_series_data[name] = df['value']
    macro_df = pd.DataFrame(all_series_data)
    macro_df.ffill(inplace=True)
    return macro_df

def get_single_symbol_data(symbol, start_time, end_time):
    """Fetches historical price data for one symbol from Coinbase."""
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("Coinbase API keys are missing.")
    client = RESTClient(api_key=api_key, api_secret=api_secret)
    response = client.get_candles(
        product_id=symbol,
        start=str(int(start_time.timestamp())),
        end=str(int(end_time.timestamp())),
        granularity="ONE_DAY"
    )
    candle_dicts = [{"start": c.start, "high": c.high, "low": c.low, "open": c.open, "close": c.close, "volume": c.volume} for c in response.candles]
    if not candle_dicts: return None
    df = pd.DataFrame(candle_dicts)
    df['time'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s')
    df.set_index('time', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.sort_index(inplace=True)
    return df

def get_top_5_market_data():
    """Fetches all data sources concurrently for maximum speed."""
    fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key: raise ValueError("FRED API Key is missing.")
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)
    all_crypto_data = {}
    
    with ThreadPoolExecutor(max_workers=len(TOP_5_SYMBOLS) + 1) as executor:
        fred_future = executor.submit(get_fred_data, start_time, end_time, fred_api_key)
        coinbase_futures = {executor.submit(get_single_symbol_data, symbol, start_time, end_time): symbol for symbol in TOP_5_SYMBOLS}
        macro_df = fred_future.result()
        for future in coinbase_futures:
            symbol = coinbase_futures[future]
            try:
                symbol_df = future.result()
                if symbol_df is not None:
                    combined_df = symbol_df.join(macro_df, how='left')
                    combined_df.ffill(inplace=True)
                    all_crypto_data[symbol] = combined_df
            except Exception as e:
                print(f"Failed to process result for {symbol}: {e}")
    return all_crypto_data

def run_simple_moving_average_backtest(symbol, data_df):
    """Runs a simple backtest strategy."""
    short_window, long_window = 10, 30
    signals = pd.DataFrame(index=data_df.index)
    signals['short_mavg'] = data_df['close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data_df['close'].rolling(window=long_window, min_periods=1).mean()
    signals['signal'] = 0.0
    signals.loc[signals.index[short_window:], 'signal'] = np.where(signals['short_mavg'].iloc[short_window:] > signals['long_mavg'].iloc[short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    initial_capital = 10000.0
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions[symbol] = 1 * signals['positions']
    portfolio = positions.multiply(data_df['close'], axis=0)
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(data_df['close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(data_df['close'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    returns = portfolio['total'].pct_change()
    results = {
        "initial_capital": initial_capital,
        "final_value": portfolio['total'].iloc[-1],
        "total_return_percent": ((portfolio['total'].iloc[-1] / initial_capital) - 1) * 100,
        "total_trades": int(abs(signals['positions']).sum()),
        "sharpe_ratio": (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() != 0 else 0
    }
    return results

# --- API Endpoints ---
@app.route('/api/data', methods=['GET'])
def get_market_data_api():
    """Serves the combined market data for the dashboard charts."""
    try:
        data_dict = get_top_5_market_data()
        json_payload = {}
        for symbol, df in data_dict.items():
            df_reset = df.reset_index()
            df_reset.rename(columns={'index': 'time'}, inplace=True)
            df_reset['time'] = df_reset['time'].dt.strftime('%Y-%m-%d')
            json_payload[symbol] = df_reset.to_dict(orient='records')
        return jsonify(json_payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-backtest', methods=['POST'])
def run_backtest_api():
    """Runs a backtest for a specific symbol provided by the frontend."""
    try:
        request_data = request.get_json()
        symbol_to_test = request_data.get('symbol')
        if not symbol_to_test or symbol_to_test not in TOP_5_SYMBOLS:
            return jsonify({"error": "A valid 'symbol' from the top 5 must be provided."}), 400
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=90)
        symbol_data_df = get_single_symbol_data(symbol_to_test, start_time, end_time)
        
        if symbol_data_df is None:
            return jsonify({"error": f"Could not fetch data for symbol {symbol_to_test}."}), 404
            
        backtest_results = run_simple_moving_average_backtest(symbol_to_test, symbol_data_df)
        return jsonify(backtest_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
