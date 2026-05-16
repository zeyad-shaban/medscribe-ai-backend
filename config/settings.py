from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pydantic automatically finds CLERK_SECRET_KEY in your .env.local
    clerk_secret_key: str
    gemini_api_key: str
    groq_secret_key: str
    next_public_clerk_publishable_key: str
    smtp_password: str
    smtp_user: str
    vercel_oidc_token: str
    
    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8")

@lru_cache
def get_settings() -> Settings:
    return Settings() # type: ignore