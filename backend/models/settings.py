# backend/models/settings.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class UserSettings(Base):
    __tablename__ = "user_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    
    trading_mode = Column(String, default="paper", nullable=False)
    training_strategy = Column(String, default="static", nullable=False)
    max_daily_drawdown_pct = Column(Float, default=5.0, nullable=False)

    paper_api_key = Column(String, nullable=True, default="")
    paper_api_secret = Column(String, nullable=True, default="")
    live_api_key = Column(String, nullable=True, default="")
    live_api_secret = Column(String, nullable=True, default="")
    
    default_symbol = Column(String, default="BTC/USDT")
    default_strategy = Column(String, default="basic")
    
    owner = relationship("User", back_populates="settings")
