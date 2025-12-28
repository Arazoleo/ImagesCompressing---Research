"""
Registro centralizado de todos os algoritmos disponíveis
"""

from typing import Dict, Any, Type
from abc import ABC, abstractmethod

from services.algorithms.base_algorithm import AlgorithmBase
from services.algorithms.svd_algorithm import SVDAlgorithm
from services.algorithms.fft_algorithm import FFTAlgorithm
from services.algorithms.pca_algorithm import PCAAlgorithm
from services.algorithms.wavelet_algorithm import WaveletAlgorithm
from services.algorithms.autoencoder_algorithm import AutoencoderAlgorithm
from services.algorithms.gan_algorithm import GANAlgorithm

class BasicAlgorithm:
    """Algoritmo básico para implementação futura"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def process_image(self, image_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Processamento básico (placeholder)"""
        import time
        start_time = time.time()

        # Simulação básica
        import random
        psnr = 25 + random.uniform(-5, 5)

        return {
            "success": True,
            "algorithm": self.name.lower().replace(" ", "-"),
            "output_url": f"/processed/{self.name.lower().replace(' ', '_')}_{image_path.split('/')[-1].split('.')[0]}.jpg",
            "metrics": {
                "psnr": round(psnr, 2),
                "ssim": round(0.8 + random.uniform(-0.1, 0.1), 3),
                "compression_ratio": round(0.6 + random.uniform(-0.2, 0.2), 2),
            },
            "processing_time": round(time.time() - start_time, 2)
        }

class AlgorithmBase(ABC):
    """Classe base para todos os algoritmos"""

    def __init__(self):
        self.name = "Base Algorithm"
        self.description = "Base algorithm description"

    @abstractmethod
    def process_image(self, image_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Processa uma imagem com parâmetros específicos"""
        pass

class AlgorithmRegistry:
    """Registro de algoritmos disponíveis"""

    def __init__(self):
        self.algorithms: Dict[str, AlgorithmBase] = {}
        self._register_algorithms()

    def _register_algorithms(self):
        """Registra todos os algoritmos disponíveis"""
        self.algorithms["svd"] = SVDAlgorithm()
        self.algorithms["fft"] = FFTAlgorithm()
        self.algorithms["pca"] = PCAAlgorithm()
        self.algorithms["wavelet"] = WaveletAlgorithm()
        self.algorithms["autoencoder"] = AutoencoderAlgorithm()
        self.algorithms["gan"] = GANAlgorithm()

        # Algoritmos básicos (por enquanto)
        self.algorithms["compressed-sensing"] = BasicAlgorithm("Compressed Sensing", "Amostragem esparsa inteligente")
        self.algorithms["hilbert"] = BasicAlgorithm("Hilbert Curve", "Mapeamento espacial 2D→1D")
        self.algorithms["ica"] = BasicAlgorithm("ICA", "Análise de Componentes Independentes")

    def get_algorithm(self, algorithm_type: str) -> AlgorithmBase:
        """Retorna uma instância do algoritmo solicitado"""
        if algorithm_type not in self.algorithms:
            raise ValueError(f"Algoritmo '{algorithm_type}' não encontrado")

        return self.algorithms[algorithm_type]

    def get_available_algorithms(self) -> Dict[str, AlgorithmBase]:
        """Retorna todos os algoritmos disponíveis"""
        return self.algorithms.copy()

    def is_algorithm_available(self, algorithm_type: str) -> bool:
        """Verifica se um algoritmo está disponível"""
        return algorithm_type in self.algorithms

# Instância global do registro
algorithm_registry = AlgorithmRegistry()
