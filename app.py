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

# --- UPGRADE: New function to dynamically create macro data ---
def generate_macro_data(start_date, end_date):
    """
    Creates a DataFrame with simulated macro data for a given date range.
    This replaces the need for us_macro.csv.
    """
    print(f"Generating macro data from {start_date} to {end_date}...")
    # Create a full date range for the period
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    macro_df = pd.DataFrame(index=date_range)

    # Simulate the data. We'll start with known values and forward-fill them,
    # as these metrics don't typically change daily.
    macro_df['fed_funds_rate'] = 5.25 # Last known value
    macro_df['cpi'] = 306.3 # Last known value

    # Forward-fill any potential gaps (though there shouldn't be any here)
    macro_df.ffill(inplace=True)
    
    print("✅ Macro data generated successfully.")
    return macro_df

# --- Data Fetching Functions ---
def get_single_symbol_data(symbol, start_time, end_time):
    """Fetches and processes data for ONE specified symbol for a given timeframe."""
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    if not api_key: raise ValueError("Coinbase API Key is missing.")

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
    """Fetches data for all top 5 symbols and merges with generated macro data."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)

    # UPGRADE: Generate macro data dynamically instead of reading a file
    macro_df = generate_macro_data(start_date=start_time, end_date=end_time)
    
    all_crypto_data = {}
    for symbol in TOP_5_SYMBOLS:
        print(f"Fetching market data for {symbol}...")
        try:
            symbol_df = get_single_symbol_data(symbol, start_time, end_time)
            if symbol_df is not None:
                # This join will now work perfectly as the date ranges match
                combined_df = symbol_df.join(macro_df, how='left')
                combined_df.ffill(inplace=True) # Fill any weekend gaps in macro data
                
                all_crypto_data[symbol] = combined_df
                print(f"✅ Successfully processed {symbol}.")
        except Exception as e:
            print(f"❌ Failed to fetch or process {symbol}: {e}")
    return all_crypto_data

# --- Backtesting Logic (no changes needed) ---
def run_simple_moving_average_backtest(symbol, data_df):
    # ... (This function remains the same as the last version)
    return results

# --- API Endpoints (no changes needed to the endpoint logic) ---
@app.route('/api/data', methods=['GET'])
def get_market_data_api():
    # ... (This function remains the same)
    pass

@app.route('/api/run-backtest', methods=['POST'])
def run_backtest_api():
    # ... (This function remains the same)
    pass

# Helper functions to re-add to the end for completeness
def run_simple_moving_average_backtest(symbol, data_df):
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

@app.route('/api/data', methods=['GET'])
def get_market_data_api():
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
