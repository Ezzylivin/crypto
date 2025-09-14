# backend/schemas/analytics.py
from pydantic import BaseModel
from datetime import datetime

class TradeHistoryOut(BaseModel):
    id: int
    symbol: str
    entry_reason: str | None
    entry_price: float
    entry_timestamp: datetime
    exit_price: float | None
    exit_timestamp: datetime | None
    profit_loss_pct: float | None

    class Config:
        from_attributes = True
