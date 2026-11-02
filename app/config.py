from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Quant Harbor"
    database_url: str = "sqlite:///./data/trader.db"
    admin_token: str = "local-dev-token"
    trading_master_key: str | None = None
    allow_live_trading: bool = False

    @property
    def encryption_secret(self) -> str:
        return self.trading_master_key or self.admin_token


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

