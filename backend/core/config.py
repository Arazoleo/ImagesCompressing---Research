"""
Configurações centrais da aplicação
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configurações da aplicação"""

    # API
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Server
    SERVER_NAME: str = "ImageStudio API"
    SERVER_HOST: str = "http://localhost"
    DEBUG: bool = True

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "processed"
    TEMP_DIR: Path = BASE_DIR / "temp"

    # Image processing
    MAX_IMAGE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
    MAX_IMAGES_PER_REQUEST: int = 10

    # Processing
    MAX_PROCESSING_TIME: int = 300  # 5 minutes
    CONCURRENT_PROCESSES: int = 3

    # Julia/Python integration
    JULIA_EXECUTABLE: str = "julia"
    PYTHON_SCRIPTS_DIR: Path = BASE_DIR.parent / "CompressedSensing"
    SVD_SCRIPTS_DIR: Path = BASE_DIR.parent / "CompressorSVD"

    class Config:
        env_file = ".env"
        case_sensitive = True

# Instância global das configurações
settings = Settings()
