import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from coinbase.rest import RESTClient

app = Flask(__name__)
CORS(app)

TOP_5_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]

# --- Data Fetching Functions ---

def get_single_symbol_data(symbol):
    """Fetches and processes data for ONE specified symbol."""
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    if not api_key: raise ValueError("Coinbase API Key is missing.")

    client = RESTClient(api_key=api_key, api_secret=api_secret)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)
    
    response = client.get_candles(
        product_id=symbol,
        start=str(int(start_time.timestamp())),
        end=str(int(end_time.timestamp())),
        granularity="ONE_DAY"
    )
    
    candle_dicts = [{"start": c.start, "high": c.high, "low": c.low, "open": c.open, "close": c.close, "volume": c.volume} for c in response.candles]
    if not candle_dicts:
        return None

    df = pd.DataFrame(candle_dicts)
    # UPGRADE: Convert index to datetime objects for proper joining
    df['time'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s')
    df.set_index('time', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.sort_index(inplace=True)
    
    return df

def get_top_5_market_data():
    """Fetches data for all top 5 symbols for the dashboard charts."""
    macro_df = pd.read_csv("us_macro.csv", index_col='date', parse_dates=True)
    
    all_crypto_data = {}
    for symbol in TOP_5_SYMBOLS:
        print(f"Fetching market data for {symbol}...")
        try:
            symbol_df = get_single_symbol_data(symbol)
            if symbol_df is not None:
                # UPGRADE: Use a 'left' join. This keeps all crypto data and adds macro data where dates match.
                # For dates without macro data, the new columns will be NaN (Not a Number).
                combined_df = symbol_df.join(macro_df, how='left')
                # UPGRADE: Forward-fill missing macro data to avoid gaps in the chart
                combined_df[['fed_funds_rate', 'cpi']] = combined_df[['fed_funds_rate', 'cpi']].fillna(method='ffill')
                
                all_crypto_data[symbol] = combined_df
                print(f"✅ Successfully processed {symbol}. Shape after join: {combined_df.shape}")
        except Exception as e:
            print(f"❌ Failed to fetch or process {symbol}: {e}")
    return all_crypto_data

# --- Backtesting Logic ---

def run_simple_moving_average_backtest(symbol, data_df):
    """Upgraded with modern pandas/numpy practices."""
    short_window = 10
    long_window = 30

    signals = pd.DataFrame(index=data_df.index)
    signals['short_mavg'] = data_df['close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data_df['close'].rolling(window=long_window, min_periods=1).mean()
    
    # UPGRADE: Use .loc for safer assignment to prevent warnings
    signals['signal'] = 0.0
    signals.loc[signals.index[short_window:], 'signal'] = np.where(
        signals['short_mavg'].iloc[short_window:] > signals['long_mavg'].iloc[short_window:], 1.0, 0.0
    )
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
        # UPGRADE: The index is now a datetime object, convert to string for JSON
        json_payload = {
            symbol: df.reset_index().rename(columns={'index': 'time'}).to_dict(orient='records')
            for symbol, df in data_dict.items()
        }
        # Convert datetime objects to ISO format strings
        for symbol in json_payload:
            for record in json_payload[symbol]:
                record['time'] = record['time'].isoformat()

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
        
        symbol_data_df = get_single_symbol_data(symbol_to_test)
        if symbol_data_df is None:
            return jsonify({"error": f"Could not fetch data for symbol {symbol_to_test}."}), 404

        backtest_results = run_simple_moving_average_backtest(symbol_to_test, symbol_data_df)
        return jsonify(backtest_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
