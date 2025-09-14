# backend/services/storage_service.py
import os
import boto3
import pandas as pd
import joblib
from io import StringIO, BytesIO
from core.config import settings
from botocore.exceptions import ClientError

class StorageService:
    def __init__(self):
        self.is_production = settings.APP_ENV == "production"
        if self.is_production:
            try:
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=settings.S3_ENDPOINT_URL,
                    aws_access_key_id=settings.S3_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
                )
                self.bucket_name = settings.S3_BUCKET_NAME
                print("INFO: StorageService initialized in PRODUCTION mode.")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize S3 client: {e}")
        else:
            print("INFO: StorageService initialized in DEVELOPMENT mode.")
            self.local_data_dir = "user_data"

    def _get_cloud_path(self, user_id: int, folder: str, filename: str) -> str:
        return f"user_data/{user_id}/{folder}/{filename}"

    def _get_local_path(self, user_id: int, folder: str, filename: str) -> str:
        path = os.path.join(self.local_data_dir, str(user_id), folder)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, filename)

    def save_df_as_csv(self, df: pd.DataFrame, user_id: int, filename: str):
        if self.is_production:
            key = self._get_cloud_path(user_id, "data", filename)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=csv_buffer.getvalue())
        else:
            filepath = self._get_local_path(user_id, "data", filename)
            df.to_csv(filepath, index=False)

    def read_csv_as_df(self, user_id: int, filename: str) -> pd.DataFrame:
        if self.is_production:
            key = self._get_cloud_path(user_id, "data", filename)
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                return pd.read_csv(response['Body'])
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    raise FileNotFoundError(f"Data file not found in cloud: {filename}")
                raise
        else:
            filepath = self._get_local_path(user_id, "data", filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found locally: {filename}")
            return pd.read_csv(filepath)

    def save_model(self, model, user_id: int, filename: str):
        if self.is_production:
            key = self._get_cloud_path(user_id, "models", filename)
            model_buffer = BytesIO()
            joblib.dump(model, model_buffer)
            model_buffer.seek(0)
            self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=model_buffer.getvalue())
        else:
            filepath = self._get_local_path(user_id, "models", filename)
            joblib.dump(model, filepath)

    def model_exists(self, user_id: int, filename: str) -> bool:
        if self.is_production:
            key = self._get_cloud_path(user_id, "models", filename)
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
                return True
            except ClientError:
                return False
        else:
            filepath = self._get_local_path(user_id, "models", filename)
            return os.path.exists(filepath)

    def list_models(self, user_id: int) -> list[str]:
        if self.is_production:
            prefix = f"user_data/{user_id}/models/"
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' not in response: return []
            return [obj['Key'].split('/')[-1] for obj in response['Contents'] if obj['Key'].endswith('.pkl')]
        else:
            path = os.path.join(self.local_data_dir, str(user_id), "models")
            if not os.path.exists(path): return []
            return [f for f in os.listdir(path) if f.endswith(".pkl")]

    def delete_model(self, user_id: int, filename: str):
        if self.is_production:
            key = self._get_cloud_path(user_id, "models", filename)
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
        else:
            filepath = self._get_local_path(user_id, "models", filename)
            if os.path.exists(filepath):
                os.remove(filepath)

storage = StorageService()
