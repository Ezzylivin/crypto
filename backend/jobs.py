# backend/jobs.py
import asyncio
from sqlalchemy.orm import Session
from database import SessionLocal
from models.user import User
from models.settings import UserSettings
from services.data_service import download_user_data
from services.model_service import train_user_models
from services.log_service import log_manager
from services.bot_service import bot_manager
from services.exchange_service import get_exchange_for_user

def scheduled_retraining_job():
    print("JOB: Running scheduled retraining job...")
    db = SessionLocal()
    try:
        users = db.query(User).join(UserSettings).filter(UserSettings.training_strategy == 'dynamic').all()
        if not users:
            print("JOB: No users with dynamic training enabled. Exiting.")
            return

        print(f"JOB: Found {len(users)} user(s) for dynamic retraining.")
        loop = asyncio.get_event_loop()
        for user in users:
            try:
                symbol = user.settings.default_symbol
                loop.run_until_complete(log_manager.log(user.id, "JOB: Starting scheduled retraining..."))
                download_user_data(user_id=user.id, exchange='binance', symbol=symbol)
                train_user_models(user_id=user.id, symbol=symbol)
                loop.run_until_complete(log_manager.log(user.id, "JOB: Scheduled retraining complete."))
            except Exception as e:
                loop.run_until_complete(log_manager.log(user.id, f"JOB-ERROR: Retraining failed: {e}"))
    finally:
        db.close()

def check_account_drawdown_job():
    print("JOB: Running circuit breaker check...")
    db = SessionLocal()
    try:
        active_uids = list(set([int(s.split(':')[0]) for s in bot_manager.running_bots.keys()]))
        if not active_uids:
            return

        for uid in active_uids:
            user = db.query(User).filter(User.id == uid).first()
            if not user or not user.settings:
                continue
            try:
                exchange = get_exchange_for_user(uid, db)
                current_bal = exchange.fetch_balance()['total']['USDT']
                balance_24h_ago = current_bal * (1 + (user.settings.max_daily_drawdown_pct + 1) / 100)
                
                drawdown_pct = ((balance_24h_ago - current_bal) / balance_24h_ago) * 100
                if drawdown_pct > user.settings.max_daily_drawdown_pct:
                    asyncio.run(log_manager.log(uid, f"CRITICAL: CIRCUIT BREAKER! Drawdown of {drawdown_pct:.2f}% exceeds limit. Stopping all bots."))
                    for sid in list(bot_manager.running_bots.keys()):
                        if sid.startswith(f"{uid}:"):
                            _, ex, sym = sid.split(':')
                            bot_manager.stop_bot(uid, ex, sym)
            except Exception as e:
                asyncio.run(log_manager.log(uid, f"JOB-ERROR: Circuit breaker check failed: {e}"))
    finally:
        db.close()
