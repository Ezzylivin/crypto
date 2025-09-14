# backend/services/chart_service.py
from datetime import datetime, timedelta
from typing import List, Dict

def generate_backtest_chart_data(returns: List[float]) -> Dict[str, List]:
    pnl_curve = []
    timestamps = []
    cumulative = 1.0
    now = datetime.utcnow()

    for i, r in enumerate(returns):
        cumulative *= (1 + r)
        pnl_curve.append(round((cumulative - 1) * 100, 4))
        timestamps.append((now + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M"))

    return {"pnl_curve": pnl_curve, "timestamps": timestamps}
