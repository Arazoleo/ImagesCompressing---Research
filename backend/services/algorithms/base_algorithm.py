"""
Classe base para todos os algoritmos de processamento de imagem
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class AlgorithmBase(ABC):
    """Classe base para todos os algoritmos"""

    def __init__(self, name: str = "Base Algorithm", description: str = "Base algorithm description"):
        self.name = name
        self.description = description

    @abstractmethod
    async def process(self, image_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Processa uma imagem com parâmetros específicos"""
        pass
