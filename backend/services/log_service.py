# backend/services/log_service.py
import asyncio
from collections import defaultdict
from fastapi import WebSocket
from typing import List, Dict

class LogManager:
    def __init__(self):
        self.subscribers: Dict[int, List[WebSocket]] = defaultdict(list)

    def register(self, user_id: int, websocket: WebSocket):
        self.subscribers[user_id].append(websocket)

    def unregister(self, user_id: int, websocket: WebSocket):
        if websocket in self.subscribers[user_id]:
            self.subscribers[user_id].remove(websocket)

    async def log(self, user_id: int, message: str):
        print(f"[USER {user_id}] {message}")
        for ws in self.subscribers.get(user_id, []):
            try:
                await ws.send_text(message)
            except Exception:
                pass

log_manager = LogManager()
