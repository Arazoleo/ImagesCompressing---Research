"""
Configuração de logging da aplicação
"""

import logging
import sys
from pathlib import Path
from core.config import settings

def setup_logging():
    """Configura o sistema de logging"""

    # Criar diretório de logs
    log_dir = settings.BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configuração do logger raiz
    logging.basicConfig(
        level=logging.INFO if not settings.DEBUG else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),

            # File handler
            logging.FileHandler(log_dir / "imagestudio.log"),
        ]
    )

    # Logger específico da aplicação
    logger = logging.getLogger("imagestudio")
    logger.setLevel(logging.INFO if not settings.DEBUG else logging.DEBUG)

    # Handler para processamento de imagens
    processing_handler = logging.FileHandler(log_dir / "processing.log")
    processing_handler.setLevel(logging.INFO)
    processing_formatter = logging.Formatter(
        '%(asctime)s - PROCESSING - %(levelname)s - %(message)s'
    )
    processing_handler.setFormatter(processing_formatter)

    # Handler para erros
    error_handler = logging.FileHandler(log_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s - ERROR - %(levelname)s - %(message)s'
    )
    error_handler.setFormatter(error_formatter)

    # Adicionar handlers ao logger
    logger.addHandler(processing_handler)
    logger.addHandler(error_handler)

    return logger

# Logger global da aplicação
logger = logging.getLogger("imagestudio")
