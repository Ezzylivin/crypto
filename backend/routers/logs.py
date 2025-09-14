# backend/routers/logs.py
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from database import get_db
from core.deps import get_current_user_ws
from services.log_service import log_manager

router = APIRouter()

@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    db: Session = next(get_db())
    user = await get_current_user_ws(websocket, db)
    db.close()
    
    if not user:
        return

    await websocket.accept()
    log_manager.register(user.id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        log_manager.unregister(user.id, websocket)
