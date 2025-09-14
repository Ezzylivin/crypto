    # backend/services/model_service.py

import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from services.storage_service import storage
from services.trading_logic_service import generate_features
from fastapi import HTTPException

def train_user_models(user_id: int, symbol: str = "BTC/USDT", exchange: str = "binance"):
    """Trains the full ensemble of models."""
    safe_symbol = symbol.replace('/', '-')
    filename = f"{exchange}_{safe_symbol}_1h.csv"
    
    try:
        df = storage.read_csv_as_df(user_id, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Data for {symbol} not found.")

    if len(df) < 200:
        raise ValueError("Not enough data. Download at least 200 periods.")

    print(f"INFO (User {user_id}): Generating features...")
    df_featured = generate_features(df.copy())
    
    df_featured.ta.bbands(length=20, append=True)
    bandwidth = df_featured['BBB_20_2.0']
    df_featured['regime'] = (bandwidth > bandwidth.quantile(0.6)).astype(int)
    df_featured['target'] = (df_featured['close'].shift(-1) > df_featured['close']).astype(int)
    df_featured.dropna(inplace=True)
    
    if df_featured.empty:
        raise ValueError("Dataframe empty after feature generation.")
        
    features_to_exclude = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'target', 'regime',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'
    ]
    all_features = [c for c in df_featured.columns if c not in features_to_exclude and 'log' not in c and 'stoch' not in c]

    X_regime, y_regime = df_featured[all_features], df_featured['regime']
    print(f"INFO (User {user_id}): Training regime model...")
    regime_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
    regime_model.fit(X_regime, y_regime)
    regime_model.feature_names_in_ = list(X_regime.columns)
    storage.save_model(regime_model, user_id, "regime_model.pkl")

    trending_data = df_featured[df_featured['regime'] == 1]
    ranging_data = df_featured[df_featured['regime'] == 0]

    if len(trending_data) < 50 or len(ranging_data) < 50:
        raise ValueError("Not enough distinct data for both regimes.")

    print(f"INFO (User {user_id}): Training trending specialist...")
    X_trend, y_trend = trending_data[all_features], trending_data['target']
    trending_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    trending_model.fit(X_trend, y_trend)
    trending_model.feature_names_in_ = list(X_trend.columns)
    storage.save_model(trending_model, user_id, "trending_model.pkl")

    print(f"INFO (User {user_id}): Training ranging specialist...")
    X_range, y_range = ranging_data[all_features], ranging_data['target']
    ranging_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    ranging_model.fit(X_range, y_range)
    ranging_model.feature_names_in_ = list(X_range.columns)
    storage.save_model(ranging_model, user_id, "ranging_model.pkl")

    print(f"INFO (User {user_id}): All models trained successfully.")
    return ["regime_model.pkl", "trending_model.pkl", "ranging_model.pkl"]

def get_user_models(user_id: int) -> list[str]:
    return storage.list_models(user_id)

def delete_user_model(user_id: int, filename: str) -> bool:
    if ".." in filename or "/" in filename:
        return False
    storage.delete_model(user_id, filename)
    return True
