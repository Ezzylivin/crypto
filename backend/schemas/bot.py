# backend/schemas/bot.py
from pydantic import BaseModel

class BotRequest(BaseModel):
    exchange: str
    symbol: str
    strategy: str = "default"
