import os
from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta, timezone
from coinbase.rest import RESTClient

app = Flask(__name__)
CORS(app) # Allow requests from other origins

def get_combined_data():
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")

    if not api_key: raise ValueError("Coinbase API Key is missing.")

    client = RESTClient(api_key=api_key, api_secret=api_secret)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)

    response = client.get_candles(
        product_id="BTC-USD",
        start=str(int(start_time.timestamp())),
        end=str(int(end_time.timestamp())),
        granularity="ONE_DAY"
    )

    candle_dicts = [{"start": c.start, "high": c.high, "low": c.low, "open": c.open, "close": c.close, "volume": c.volume} for c in response.candles]
    btc_df = pd.DataFrame(candle_dicts)
    btc_df['time'] = pd.to_datetime(pd.to_numeric(btc_df['start']), unit='s').dt.date
    btc_df.set_index('time', inplace=True)
    btc_df[['open', 'high', 'low', 'close', 'volume']] = btc_df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    btc_df.sort_index(inplace=True)

    macro_df = pd.read_csv("us_macro.csv", index_col='date', parse_dates=True)
    macro_df.index = macro_df.index.date

    combined_df = btc_df.join(macro_df, how='inner')
    return combined_df

# --- New Backtesting Logic ---
def run_simple_moving_average_backtest(btc_df):
    """
    A simple example of a backtesting strategy.
    Strategy: Buy when the short-term moving average crosses above the long-term one. Sell when it crosses below.
    """
    short_window = 10  # 10 days
    long_window = 30   # 30 days

    # Calculate moving averages
    signals = pd.DataFrame(index=btc_df.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = btc_df['close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = btc_df['close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][short_window:] = \
        pd.np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    # --- Calculate simple performance metrics ---
    initial_capital = float(10000.0)
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['BTC'] = 1 * signals['positions'] # Buy 1 BTC
    
    portfolio = positions.multiply(btc_df['close'], axis=0)
    pos_diff = positions.diff()

    portfolio['holdings'] = (positions.multiply(btc_df['close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(btc_df['close'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    
    returns = portfolio['total'].pct_change()

    # --- Prepare Results ---
    results = {
        "initial_capital": initial_capital,
        "final_value": portfolio['total'][-1],
        "total_return_percent": ((portfolio['total'][-1] / initial_capital) - 1) * 100,
        "total_trades": int(abs(signals['positions']).sum()),
        "sharpe_ratio": returns.mean() / returns.std() * pd.np.sqrt(365) # Annualized Sharpe Ratio
    }
    
    return results

# --- New Backtesting API Endpoint ---
@app.route('/api/run-backtest', methods=['POST'])
def run_backtest_api():
    """
    Runs a backtest for a given strategy.
    For now, it only runs one strategy, but it can be expanded.
    """
    try:
        # We need the historical data to run the backtest on
        btc_data_df = get_combined_data() # We reuse our existing data function
        
        # Here you could check request.json['strategy'] to run different functions
        backtest_results = run_simple_moving_average_backtest(btc_data_df)
        
        return jsonify(backtest_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/data')
def get_data_api():
    try:
        data = get_combined_data()
        return jsonify(data.reset_index().to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
