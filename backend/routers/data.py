# backend/routers/data.py
from fastapi import APIRouter, Depends, BackgroundTasks
from schemas.data import DownloadRequest
from models.user import User
from core.deps import get_current_user
from services.data_service import download_user_data
from services.log_service import log_manager
import asyncio

router = APIRouter()

def download_task(user_id: int, exchange: str, symbol: str):
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(log_manager.log(user_id, f"INFO: Starting download for {symbol}..."))
        download_user_data(user_id, exchange, symbol)
        loop.run_until_complete(log_manager.log(user_id, f"INFO: Download completed for {symbol}."))
    except Exception as e:
        loop.run_until_complete(log_manager.log(user_id, f"ERROR: Download failed for {symbol}: {e}"))

@router.post("/download-data")
async def download_data_route(req: DownloadRequest, background_tasks: BackgroundTasks, user: User = Depends(get_current_user)):
    background_tasks.add_task(download_task, user.id, req.exchange, req.symbol)
    return {"message": f"Download initiated for {req.symbol}. Check logs for status."}
