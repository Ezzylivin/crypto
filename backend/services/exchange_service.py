# backend/services/exchange_service.py
import ccxt
from fastapi import HTTPException
from sqlalchemy.orm import Session
from models.settings import UserSettings

def get_exchange_for_user(user_id: int, db: Session, exchange_id: str = 'binance') -> ccxt.Exchange:
    settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
    if not settings:
        raise HTTPException(status_code=404, detail="User settings not found.")

    mode = settings.trading_mode
    api_key = settings.live_api_key if mode == 'live' else settings.paper_api_key
    secret = settings.live_api_secret if mode == 'live' else settings.paper_api_secret

    if mode == 'live' and (not api_key or not secret):
        raise HTTPException(status_code=400, detail="Live API credentials are not set.")

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'apiKey': api_key, 'secret': secret})
        
        if mode == 'paper':
            exchange.set_sandbox_mode(True)
        
        return exchange
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize exchange '{exchange_id}': {e}")
