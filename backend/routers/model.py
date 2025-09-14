# backend/routers/model.py
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from typing import List
from models.user import User
from core.deps import get_current_user
from services.model_service import train_user_models, get_user_models, delete_user_model
from services.log_service import log_manager
import asyncio

router = APIRouter()

def train_task(user_id: int):
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(log_manager.log(user_id, "INFO: Starting ensemble model training..."))
        train_user_models(user_id)
        loop.run_until_complete(log_manager.log(user_id, "INFO: Ensemble model training completed successfully."))
    except Exception as e:
        loop.run_until_complete(log_manager.log(user_id, f"ERROR: Model training failed: {e}"))

@router.post("/train-model")
async def train_model_route(background_tasks: BackgroundTasks, user: User = Depends(get_current_user)):
    background_tasks.add_task(train_task, user.id)
    return {"message": "Ensemble model training started. Check logs for progress."}

@router.get("/models", response_model=List[str])
def list_models_route(user: User = Depends(get_current_user)):
    return get_user_models(user.id)

@router.delete("/models/{filename}")
def delete_model_route(filename: str, user: User = Depends(get_current_user)):
    if not delete_user_model(user.id, filename):
        raise HTTPException(status_code=404, detail="Model not found or access denied.")
    return {"detail": "Model deleted successfully."}
