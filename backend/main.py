# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from apscheduler.schedulers.background import BackgroundScheduler
from routers import auth, data, model, backtest, bot, logs, settings, analytics
from jobs import scheduled_retraining_job, check_account_drawdown_job

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Trading Bot API",
    description="A comprehensive API for managing, training, and running an AI-driven trading bot."
)

@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(scheduled_retraining_job, 'interval', days=5, id="retraining_job")
    scheduler.add_job(check_account_drawdown_job, 'interval', minutes=15, id="circuit_breaker_job")
    scheduler.start()
    print("INFO: Background scheduler started with all jobs.")

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(settings.router, prefix="/settings", tags=["User Settings"])
app.include_router(data.router, tags=["Data Management"])
app.include_router(model.router, tags=["AI Model"])
app.include_router(backtest.router, tags=["Backtesting"])
app.include_router(bot.router, prefix="/bot", tags=["Trading Bot"])
app.include_router(logs.router, tags=["Real-time Logs"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the AI Trading Bot API"}
