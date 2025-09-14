# backend/services/data_service.py
import ccxt
import pandas as pd
from services.storage_service import storage
from fastapi import HTTPException

def download_user_data(user_id: int, exchange: str, symbol: str, timeframe: str = '1h', limit: int = 1000):
    safe_symbol = symbol.replace('/', '-')
    filename = f"{exchange}_{safe_symbol}_{timeframe}.csv"
    
    try:
        exchange_class = getattr(ccxt, exchange)()
        if not exchange_class.has['fetchOHLCV']:
            raise HTTPException(status_code=400, detail=f"{exchange} does not support fetching OHLCV data.")
        
        ohlcv = exchange_class.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        storage.save_df_as_csv(df, user_id, filename)
        return filename
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=504, detail=f"Network error while fetching data: {e}")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=400, detail=f"Exchange error for symbol {symbol}: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
