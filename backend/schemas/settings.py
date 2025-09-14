# backend/schemas/settings.py
from pydantic import BaseModel
from typing import Literal

class UserSettingsIn(BaseModel):
    paper_api_key: str | None = None
    paper_api_secret: str | None = None
    live_api_key: str | None = None
    live_api_secret: str | None = None
    default_symbol: str | None = None
    max_daily_drawdown_pct: float | None = None

class UserSettingsOut(UserSettingsIn):
    trading_mode: str
    training_strategy: str

    class Config:
        from_attributes = True

class TradingModeIn(BaseModel):
    mode: Literal['paper', 'live']

class TradingModeOut(BaseModel):
    mode: str

class TrainingStrategyIn(BaseModel):
    strategy: Literal['static', 'dynamic']

class TrainingStrategyOut(BaseModel):
    strategy: str
