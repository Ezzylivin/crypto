# backend/services/backtest_service.py
import pandas as pd
from services.storage_service import storage
from services.trading_logic_service import get_ensemble_prediction
from services.chart_service import generate_backtest_chart_data
from fastapi import HTTPException

TAKER_FEE = 0.001 
SLIPPAGE = 0.0005 

def run_backtest(user_id: int, exchange: str, symbol: str):
    safe_symbol = symbol.replace('/', '-')
    filename = f"{exchange}_{safe_symbol}_1h.csv"
    
    try:
        df = storage.read_csv_as_df(user_id, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Data for {symbol} not found.")
    
    if len(df) < 50:
        raise ValueError("Data file too short for a meaningful backtest.")

    in_position, balance, trades, wins = False, 1000.0, 0, 0
    entry_price = 0
    returns_log = []

    for i in range(50, len(df)):
        current_data_slice = df.iloc[:i]
        
        try:
            prediction, confidence = get_ensemble_prediction(user_id, current_data_slice.copy())
        except Exception:
            continue

        current_price = df.iloc[i]['close']

        if prediction == 1 and not in_position and confidence > 0.70:
            in_position = True
            entry_price = current_price * (1 + SLIPPAGE)
            balance -= (balance * TAKER_FEE)
            trades += 1
        
        elif prediction == 0 and in_position:
            exit_price = current_price * (1 - SLIPPAGE)
            trade_return = (exit_price - entry_price) / entry_price
            returns_log.append(trade_return)
            
            if trade_return > 0:
                wins += 1
            
            balance *= (1 + trade_return)
            balance -= (balance * TAKER_FEE)
            in_position, entry_price = False, 0

    total_return_pct = ((balance / 1000.0) - 1) * 100
    win_rate_pct = (wins / trades * 100) if trades > 0 else 0
    chart_data = generate_backtest_chart_data(returns_log)

    return {"total_return": total_return_pct, "trades": trades, "win_rate": win_rate_pct, **chart_data}
