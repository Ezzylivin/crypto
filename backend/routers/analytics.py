# backend/routers/analytics.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from models.user import User
from models.trade_history import TradeHistory
from core.deps import get_current_user
from database import get_db
from schemas.analytics import TradeHistoryOut

router = APIRouter()

@router.get("/trade-history", response_model=List[TradeHistoryOut])
def get_trade_history(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    trade_history = db.query(TradeHistory).filter(TradeHistory.user_id == user.id).order_by(TradeHistory.entry_timestamp.desc()).all()
    return trade_history
