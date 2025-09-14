# backend/schemas/data.py
from pydantic import BaseModel

class DownloadRequest(BaseModel):
    exchange: str
    symbol: str
