# backend/services/onchain_service.py

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

# --- Caching Mechanism ---
# This prevents us from hitting the API on every single bot loop or backtest run.
# The cache will hold the data for 6 hours before fetching a fresh copy.
_cache = None
_cache_timestamp = None
CACHE_DURATION_HOURS = 6

def get_onchain_data() -> pd.DataFrame:
    """
    Fetches the historical Fear & Greed Index data from the Alternative.me API.
    The data is cached in memory to avoid excessive API calls.
    
    Returns:
        pd.DataFrame: A DataFrame with 'timestamp' and 'fear_greed_index' columns.
    """
    global _cache, _cache_timestamp

    # 1. Check if a valid cache exists
    if _cache is not None and _cache_timestamp is not None:
        if datetime.now(timezone.utc) - _cache_timestamp < timedelta(hours=CACHE_DURATION_HOURS):
            print("INFO: Returning cached on-chain data.")
            return _cache

    # 2. If no valid cache, fetch from the API
    print("INFO: Fetching new on-chain data from live API...")
    try:
        # The limit=0 parameter fetches all available historical data
        url = "https://api.alternative.me/fng/?limit=0"
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        data = response.json()['data']
        
        # 3. Process the raw API data into a clean DataFrame
        df = pd.DataFrame(data)
        
        # Convert the 'value' column to a numeric type for the model
        df['value'] = pd.to_numeric(df['value'])
        
        # Rename the 'value' column to be more descriptive
        df.rename(columns={'value': 'fear_greed_index'}, inplace=True)
        
        # Convert the Unix timestamp string to a proper datetime object
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Make the timestamp timezone-aware (UTC) to allow for clean merges
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # The API returns oldest first, so we reverse it to have the newest data last
        df = df.iloc[::-1].reset_index(drop=True)

        # Select only the columns we need
        final_df = df[['timestamp', 'fear_greed_index']]

        # 4. Update the cache
        _cache = final_df
        _cache_timestamp = datetime.now(timezone.utc)
        
        return final_df

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch Fear & Greed data: {e}")
        # If the API fails but we have old cached data, return that as a fallback.
        if _cache is not None:
            print("WARNING: API fetch failed. Returning stale cached data as a fallback.")
            return _cache
        # If there's no cache at all, return an empty DataFrame
        return pd.DataFrame(columns=['timestamp', 'fear_greed_index'])
