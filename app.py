import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np # Import numpy for backtesting calculations
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
    df['time'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s').dt.date
    df.set_index('time', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.sort_index(inplace=True)
    
    return df

def get_top_5_market_data():
    """Fetches data for all top 5 symbols for the dashboard charts."""
    macro_df = pd.read_csv("us_macro.csv", index_col='date', parse_dates=True)
    macro_df.index = macro_df.index.date
    
    all_crypto_data = {}
    for symbol in TOP_5_SYMBOLS:
        print(f"Fetching market data for {symbol}...")
        try:
            symbol_df = get_single_symbol_data(symbol)
            if symbol_df is not None:
                combined_df = symbol_df.join(macro_df, how='inner')
                all_crypto_data[symbol] = combined_df
                print(f"✅ Successfully processed {symbol}.")
        except Exception as e:
            print(f"❌ Failed to fetch or process {symbol}: {e}")
    return all_crypto_data

# --- Backtesting Logic ---

def run_simple_moving_average_backtest(symbol, data_df):
    """Upgraded to be symbol-agnostic."""
    short_window = 10
    long_window = 30

    signals = pd.DataFrame(index=data_df.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = data_df['close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data_df['close'].rolling(window=long_window, min_periods=1).mean()
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()

    initial_capital = 10000.0
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions[symbol] = 1 * signals['positions'] # Buy 1 unit of the symbol
    
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
        json_payload = {
            symbol: df.reset_index().to_dict(orient='records')
            for symbol, df in data_dict.items()
        }
        return jsonify(json_payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-backtest', methods=['POST'])
def run_backtest_api():
    """Runs a backtest for a specific symbol provided by the frontend."""
    try:
        request_data = request.get_json()
        symbol_to_test = request_data.get('symbol')

        if not symbol_to_test:
            return jsonify({"error": "A 'symbol' must be provided to run a backtest."}), 400
        
        # Fetch data only for the requested symbol
        symbol_data_df = get_single_symbol_data(symbol_to_test)
        if symbol_data_df is None:
            return jsonify({"error": f"Could not fetch data for symbol {symbol_to_test}."}), 404

        # Run the backtest and return the results
        backtest_results = run_simple_moving_average_backtest(symbol_to_test, symbol_data_df)
        return jsonify(backtest_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
