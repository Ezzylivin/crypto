# backend/routers/backtest.py
from fastapi import APIRouter, Depends, HTTPException
from models.user import User
from core.deps import get_current_user
from schemas.backtest import BacktestRequest, BacktestResponse
from services.backtest_service import run_backtest

router = APIRouter()

@router.post("/backtest", response_model=BacktestResponse)
def backtest_model_route(req: BacktestRequest, user: User = Depends(get_current_user)):
    try:
        results = run_backtest(user.id, req.exchange, req.symbol)
        return results
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
