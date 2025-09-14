# backend/services/trading_logic_service.py

import pandas as pd
import pandas_ta as ta
import joblib
from io import BytesIO
from services.storage_service import storage
from services.onchain_service import get_onchain_data
from fastapi import HTTPException
from botocore.exceptions import ClientError
from sklearn.ensemble import RandomForestClassifier

def _load_model(user_id: int, model_name: str) -> RandomForestClassifier:
    """Helper function to load a specific model file for a user."""
    try:
        if not storage.model_exists(user_id, model_name):
            raise FileNotFoundError

        if storage.is_production:
            cloud_path = storage._get_cloud_path(user_id, "models", model_name)
            model_obj = storage.s3_client.get_object(Bucket=storage.bucket_name, Key=cloud_path)
            model_bytes = model_obj['Body'].read()
            model_buffer = BytesIO(model_bytes)
            model = joblib.load(model_buffer)
        else:
            local_path = storage._get_local_path(user_id, "models", model_name)
            model = joblib.load(local_path)
        return model

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Please train all models first.")
    except ClientError as e:
        if 'NoSuchKey' in str(e):
             raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Train models first.")
        raise HTTPException(status_code=500, detail=f"Cloud storage error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model '{model_name}': {e}")

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates trading features, including technical indicators and the live Fear & Greed Index."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

    df.ta.strategy("common")
    df['price_change'] = df['close'].diff()

    # Fetch the live Fear & Greed data
    onchain_df = get_onchain_data()
    if onchain_df.empty:
        # If the API fails, create a placeholder column to avoid errors
        df['fear_greed_index'] = 0
    else:
        if onchain_df['timestamp'].dt.tz is None:
            onchain_df['timestamp'] = onchain_df['timestamp'].dt.tz_localize('UTC')

        # Merge the on-chain data into the main dataframe based on the nearest previous date
        df = pd.merge_asof(df.sort_values('timestamp'), onchain_df.sort_values('timestamp'), on='timestamp')
    
    # Clean up any missing values after merging
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def get_ensemble_prediction(user_id: int, df_slice: pd.DataFrame) -> tuple[int, float]:
    """Main prediction function using the regime-switching ensemble model."""
    regime_model = _load_model(user_id, "regime_model.pkl")
    trending_model = _load_model(user_id, "trending_model.pkl")
    ranging_model = _load_model(user_id, "ranging_model.pkl")

    df_with_features = generate_features(df_slice.copy())
    
    regime_features = regime_model.feature_names_in_
    if not all(f in df_with_features.columns for f in regime_features):
        raise ValueError("Feature mismatch for regime model. Retrain.")
    latest_regime_data = df_with_features[regime_features].iloc[-1:]
    current_regime = regime_model.predict(latest_regime_data)[0]

    specialist_model = trending_model if current_regime == 1 else ranging_model
    
    prediction_features = specialist_model.feature_names_in_
    if not all(f in df_with_features.columns for f in prediction_features):
        raise ValueError("Feature mismatch for specialist model. Retrain.")
    latest_prediction_data = df_with_features[prediction_features].iloc[-1:]
    
    prediction = specialist_model.predict(latest_prediction_data)[0]
    confidence = max(specialist_model.predict_proba(latest_prediction_data)[0])

    return int(prediction), float(confidence)
