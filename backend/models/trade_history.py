# backend/models/trade_history.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from database import Base
import datetime

class TradeHistory(Base):
    __tablename__ = "trade_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    symbol = Column(String, nullable=False)
    entry_reason = Column(String, nullable=True)
    
    entry_price = Column(Float, nullable=False)
    entry_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    exit_price = Column(Float, nullable=True)
    exit_timestamp = Column(DateTime, nullable=True)
    
    profit_loss_pct = Column(Float, nullable=True)
