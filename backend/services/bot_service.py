# backend/services/bot_service.py
import threading, time, asyncio, pandas as pd, pandas_ta as ta, ccxt
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from ccxt.base.errors import NetworkError, ExchangeError
from models.trade_history import TradeHistory
from services.log_service import log_manager
from database import SessionLocal
from services.exchange_service import get_exchange_for_user
from services.trading_logic_service import get_ensemble_prediction

def _get_sleep_duration(timeframe: str) -> int:
    try:
        tf_seconds = ccxt.Exchange.parse_timeframe(timeframe)
        now_ts = int(datetime.now(timezone.utc).timestamp())
        next_candle_ts = (now_ts - (now_ts % tf_seconds)) + tf_seconds
        return max((next_candle_ts - now_ts) + 5, 5)
    except:
        return 900

class BotManager:
    def __init__(self):
        self.running_bots: dict[str, threading.Thread] = {}
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def _bot_loop(self, user_id: int, exchange_id: str, symbol: str, strategy: str):
        session_id = f"{user_id}:{exchange_id}:{symbol}"
        db: Session = SessionLocal() 
        position_held, entry_price, timeframe = False, 0.0, '1h'
        active_trade_id = None
        ATR_PERIOD, ATR_MULTIPLIER, CONF_THRESH = 14, 2.0, 0.70

        try:
            exchange = get_exchange_for_user(user_id, db, exchange_id)
            mode = "PAPER" if exchange.sandbox else "LIVE"
            async def log(msg):
                await log_manager.log(user_id, f"[{symbol} - {mode}] {msg}")
            self.loop.run_until_complete(log("INFO: Advanced Bot started."))

            while session_id in self.running_bots:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
                    if not ohlcv:
                        self.loop.run_until_complete(log("WARN: No OHLCV data. Retrying.")); time.sleep(60); continue
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    price = df['close'].iloc[-1]
                    
                    if position_held and active_trade_id:
                        df.ta.atr(length=ATR_PERIOD, append=True)
                        sl_price = entry_price - (df[f'ATRr_{ATR_PERIOD}'].iloc[-1] * ATR_MULTIPLIER)
                        self.loop.run_until_complete(log(f"INFO: Pos held. Entry:${entry_price:.4f}, Curr:${price:.4f}, SL:${sl_price:.4f}"))
                        if price < sl_price:
                            self.loop.run_until_complete(log(f"ACTION: STOP-LOSS @ ${price:.4f}. Selling."))
                            trade = db.query(TradeHistory).filter(TradeHistory.id == active_trade_id).first()
                            if trade:
                                trade.exit_price = price
                                trade.exit_timestamp = datetime.utcnow()
                                trade.profit_loss_pct = ((price - trade.entry_price) / trade.entry_price) * 100
                                db.commit()
                            position_held, entry_price, active_trade_id = False, 0.0, None
                            time.sleep(_get_sleep_duration(timeframe)); continue

                    pred, conf = get_ensemble_prediction(user_id, df.copy())
                    self.loop.run_until_complete(log(f"PRED: {'BUY' if pred==1 else 'SELL/HOLD'} @ {conf:.2%} conf."))

                    if pred == 1 and not position_held and conf >= CONF_THRESH:
                        scaler = (conf - CONF_THRESH) / (1.0 - CONF_THRESH)
                        trade_amt = max(5.0, 15.0 * scaler)
                        self.loop.run_until_complete(log(f"ACTION: BUY signal. Sizing to ${trade_amt:.2f}."))
                        new_trade = TradeHistory(user_id=user_id, symbol=symbol, entry_price=price, entry_reason=f"Ensemble BUY @ {conf:.1%}")
                        db.add(new_trade); db.commit(); db.refresh(new_trade)
                        active_trade_id = new_trade.id
                        position_held, entry_price = True, price
                    elif pred == 0 and position_held and active_trade_id:
                        self.loop.run_until_complete(log("ACTION: SELL signal. Closing position."))
                        trade = db.query(TradeHistory).filter(TradeHistory.id == active_trade_id).first()
                        if trade:
                            trade.exit_price = price
                            trade.exit_timestamp = datetime.utcnow()
                            trade.profit_loss_pct = ((price - trade.entry_price) / trade.entry_price) * 100
                            db.commit()
                        position_held, entry_price, active_trade_id = False, 0.0, None
                    
                    time.sleep(_get_sleep_duration(timeframe))
                except (NetworkError, ExchangeError) as e:
                    self.loop.run_until_complete(log(f"WARN: Net/Exch Error: {e}. Retrying.")); time.sleep(60)
                except Exception as e:
                    self.loop.run_until_complete(log(f"FATAL: Bot loop error: {e}")); break
        except Exception as e:
            self.loop.run_until_complete(log_manager.log(user_id, f"FATAL: Bot init error: {e}"))
        finally:
            db.close()
            if session_id in self.running_bots:
                del self.running_bots[session_id]
            self.loop.run_until_complete(log_manager.log(user_id, f"INFO: Bot for {symbol} stopped."))

    def start_bot(self, uid, ex, sym, strat):
        sid = f"{uid}:{ex}:{sym}"; t = threading.Thread(target=self._bot_loop, args=(uid,ex,sym,strat), daemon=True)
        if sid not in self.running_bots:
            self.running_bots[sid] = t
            t.start()

    def stop_bot(self, uid, ex, sym):
        sid = f"{uid}:{ex}:{sym}"
        if sid in self.running_bots:
            del self.running_bots[sid]

    def is_bot_running(self, uid, ex, sym):
        return f"{uid}:{ex}:{sym}" in self.running_bots

bot_manager = BotManager()
