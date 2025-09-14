# backend/routers/settings.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models.user import User
from models.settings import UserSettings
from schemas.settings import UserSettingsIn, UserSettingsOut, TradingModeIn, TradingModeOut, TrainingStrategyIn, TrainingStrategyOut
from core.deps import get_current_user
from database import get_db
from services.log_service import log_manager
import asyncio

router = APIRouter()

def get_or_create_settings(db: Session, user_id: int) -> UserSettings:
    settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
    if not settings:
        settings = UserSettings(user_id=user_id)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings

@router.get("", response_model=UserSettingsOut)
def get_settings(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return get_or_create_settings(db, user.id)

@router.post("", response_model=UserSettingsOut)
def update_settings(payload: UserSettingsIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    settings = get_or_create_settings(db, user.id)
    update_data = payload.dict(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(settings, key, value)
    db.commit()
    db.refresh(settings)
    return settings

@router.post("/trading-mode", response_model=TradingModeOut)
async def set_trading_mode(payload: TradingModeIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    settings = get_or_create_settings(db, user.id)
    settings.trading_mode = payload.mode
    db.commit()
    await log_manager.log(user.id, f"INFO: Trading mode switched to {payload.mode.upper()}.")
    return {"mode": settings.trading_mode}

@router.post("/training-strategy", response_model=TrainingStrategyOut)
async def set_training_strategy(payload: TrainingStrategyIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    settings = get_or_create_settings(db, user.id)
    settings.training_strategy = payload.strategy
    db.commit()
    await log_manager.log(user.id, f"INFO: AI Training Strategy switched to {payload.strategy.upper()}.")
    return {"strategy": settings.training_strategy}
