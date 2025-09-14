# backend/core/config.py
from pydantic_settings import BaseSettings
from pydantic import validator, Field

class Settings(BaseSettings):
    APP_ENV: str = Field("development")
    JWT_SECRET: str
    DATABASE_URL: str
    JWT_ALGORITHM: str = "HS256"

    S3_ENDPOINT_URL: str | None = None
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None
    S3_BUCKET_NAME: str | None = None

    @validator("S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_BUCKET_NAME", pre=True, always=True)
    def check_production_settings(cls, v, values):
        if values.get("APP_ENV") == "production" and v is None:
            raise ValueError("This setting is required in production environment")
        return v

    class Config:
        env_file = ".env"

settings = Settings()
