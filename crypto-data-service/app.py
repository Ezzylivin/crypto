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

@app.route('/api/data')
def get_data_api():
    try:
        data = get_combined_data()
        return jsonify(data.reset_index().to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
