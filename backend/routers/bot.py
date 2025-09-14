# backend/routers/bot.py
from fastapi import APIRouter, Depends, HTTPException
from models.user import User
from core.deps import get_current_user
from schemas.bot import BotRequest
from services.bot_service import bot_manager
from services.storage_service import storage

router = APIRouter()

@router.post("/start")
async def start_bot_route(req: BotRequest, user: User = Depends(get_current_user)):
    if bot_manager.is_bot_running(user.id, req.exchange, req.symbol):
        raise HTTPException(status_code=400, detail="Bot is already running for this symbol.")
    
    required_models = ["regime_model.pkl", "trending_model.pkl", "ranging_model.pkl"]
    for model_name in required_models:
        if not storage.model_exists(user.id, model_name):
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found. Please train models first.")

    bot_manager.start_bot(user.id, req.exchange, req.symbol, req.strategy)
    return {"status": "started"}

@router.post("/stop")
async def stop_bot_route(req: BotRequest, user: User = Depends(get_current_user)):
    bot_manager.stop_bot(user.id, req.exchange, req.symbol)
    return {"status": "stopped"}

@router.get("/status")
async def bot_status(exchange: str, symbol: str, user: User = Depends(get_current_user)):
    return {"running": bot_manager.is_bot_running(user.id, exchange, symbol)}
