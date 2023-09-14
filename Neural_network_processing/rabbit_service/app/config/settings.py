from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


CONFIG_DIR = Path(__file__).resolve().parent
BASE_DIR = CONFIG_DIR.parent


class BrokerSettings(BaseModel):
    url: str = Field(...)


class Settings(BaseSettings):
    debug: bool = Field(True)
    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    base_dir: Path = BASE_DIR

    class Config:
        env_prefix = 'APP_'
        env_file_encoding = 'utf-8'
        env_nested_delimiter = '__'


def init_settings(env_file: Optional[str] = None) -> Settings:
    env_file = CONFIG_DIR / (env_file or ".env")
    if env_file.exists():
        settings = Settings(_env_file=env_file)
    else:
        settings = Settings()
    return settings