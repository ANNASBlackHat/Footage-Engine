"""Configuration management via pydantic-settings."""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = "sqlite:///./footage_engine.db"

    # Storage
    STORAGE_BACKEND: Literal["local", "imagekit"] = "local"
    LOCAL_STORAGE_DIR: str = "./data/storage"
    UPLOAD_RAW_TO_STORAGE: bool = False  # If False, streams directly from source_url without uploading raw master
    UPLOAD_CHUNKS_TO_STORAGE: bool = False  # If True, slices individual scene chunks and uploads .mp4 files to storage


    # ImageKit Credentials (if using imagekit backend)
    IMAGEKIT_PUBLIC_KEY: str | None = None
    IMAGEKIT_PRIVATE_KEY: str | None = None
    IMAGEKIT_URL_ENDPOINT: str | None = None

    # Stock & Web Provider API Keys / Options
    PIXABAY_API_KEY: str | None = None
    PEXELS_API_KEY: str | None = None
    COVERR_API_KEY: str | None = None
    YOUTUBE_COOKIES: str | None = None  # Local file path, URL, or raw Netscape/base64 cookie string
    YOUTUBE_COOKIES_FROM_BROWSER: str | None = None  # e.g. 'chrome', 'firefox', 'brave', 'safari'


    # Vector Database
    VECTOR_STORE: Literal["in_memory", "zilliz"] = "in_memory"
    ZILLIZ_URI: str | None = None
    ZILLIZ_TOKEN: str | None = None
    ZILLIZ_COLLECTION_NAME: str = "footage_chunks"

    # Chunking & Preprocessing Thresholds
    CHUNK_THRESHOLD_SEC: float = 45.0
    SLIDING_WINDOW_SEC: float = 10.0
    SLIDING_OVERLAP_RATIO: float = 0.5

    # Embedding
    DEFAULT_EMBEDDING_MODEL: str = "microsoft/xclip-base-patch32"
    DEFAULT_EMBEDDING_VERSION: str = "1.0"
    EMBEDDING_DIMENSION: int = 512
    EMBEDDING_DEVICE: str = "auto"  # 'auto', 'cuda', or 'cpu'


@lru_cache()
def get_settings() -> Settings:
    return Settings()
