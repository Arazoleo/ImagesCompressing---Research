"""
Serviço para gerenciamento de algoritmos
"""

from typing import Dict, List, Any, Optional, Tuple
from models.schemas import AlgorithmType, AlgorithmConfig
from core.logging import logger

class AlgorithmService:
    """Serviço para informações sobre algoritmos"""

    def __init__(self):
        self.algorithms = self._load_algorithms()

    def _load_algorithms(self) -> Dict[str, AlgorithmConfig]:
        """Carregar configurações de todos os algoritmos disponíveis"""

        return {
            "svd": AlgorithmConfig(
                type=AlgorithmType.SVD,
                name="SVD Compression",
                description="Decomposição em Valores Singulares para compressão matricial",
                parameters={
                    "k": {
                        "name": "k",
                        "label": "Valores Singulares",
                        "type": "slider",
                        "default": 50,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "description": "Número de valores singulares para manter"
                    }
                }
            ),

            "compressed-sensing": AlgorithmConfig(
                type=AlgorithmType.COMPRESSED_SENSING,
                name="Compressed Sensing",
                description="Compressão baseada em amostragem esparsa com reconstrução L1",
                parameters={
                    "sampling_rate": {
                        "name": "sampling_rate",
                        "label": "Taxa de Amostragem",
                        "type": "slider",
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.1,
                        "description": "Porcentagem de amostras coletadas"
                    },
                    "sparsifying_basis": {
                        "name": "sparsifying_basis",
                        "label": "Base Esparsificante",
                        "type": "select",
                        "default": "dct",
                        "options": [
                            {"value": "dct", "label": "DCT (Cosseno)"},
                            {"value": "wavelet", "label": "Wavelet"},
                            {"value": "fft", "label": "FFT"}
                        ],
                        "description": "Transformada para representação esparsa"
                    }
                }
            ),

            "autoencoder": AlgorithmConfig(
                type=AlgorithmType.AUTOENCODER,
                name="Autoencoder",
                description="Rede neural autoencoder para compressão aprendida",
                parameters={
                    "latent_dim": {
                        "name": "latent_dim",
                        "label": "Dimensão Latente",
                        "type": "slider",
                        "default": 64,
                        "min": 16,
                        "max": 256,
                        "step": 16,
                        "description": "Dimensão do espaço latente"
                    },
                    "epochs": {
                        "name": "epochs",
                        "label": "Épocas",
                        "type": "number",
                        "default": 100,
                        "min": 10,
                        "max": 1000,
                        "description": "Número de épocas de treinamento"
                    }
                }
            ),

            "gan": AlgorithmConfig(
                type=AlgorithmType.GAN,
                name="Super-Resolution GAN",
                description="Redes Generativas Adversariais para aumento de resolução",
                parameters={
                    "scale_factor": {
                        "name": "scale_factor",
                        "label": "Fator de Escala",
                        "type": "select",
                        "default": 2,
                        "options": [
                            {"value": 2, "label": "2x"},
                            {"value": 4, "label": "4x"}
                        ],
                        "description": "Fator de aumento de resolução"
                    },
                    "downscale_factor": {
                        "name": "downscale_factor",
                        "label": "Fator de Redução",
                        "type": "slider",
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.1,
                        "description": "Quanto reduzir a resolução antes do super-resolution"
                    }
                }
            ),

            "hybrid": AlgorithmConfig(
                type=AlgorithmType.HYBRID,
                name="SVD + Autoencoder",
                description="Compressão híbrida combinando SVD com refinamento neural",
                parameters={
                    "svd_k": {
                        "name": "svd_k",
                        "label": "Valores Singulares SVD",
                        "type": "slider",
                        "default": 30,
                        "min": 5,
                        "max": 100,
                        "step": 5,
                        "description": "Valores singulares para pré-processamento"
                    },
                    "refinement_epochs": {
                        "name": "refinement_epochs",
                        "label": "Épocas Refinamento",
                        "type": "number",
                        "default": 50,
                        "min": 10,
                        "max": 200,
                        "description": "Épocas para refinamento neural"
                    }
                }
            ),

            "hilbert": AlgorithmConfig(
                type=AlgorithmType.HILBERT,
                name="Curva de Hilbert",
                description="Mapeamento 2D→1D usando curva de Hilbert com autoencoder",
                parameters={
                    "curve_order": {
                        "name": "curve_order",
                        "label": "Ordem da Curva",
                        "type": "slider",
                        "default": 7,
                        "min": 4,
                        "max": 10,
                        "step": 1,
                        "description": "Ordem da curva de Hilbert (2^ordem = tamanho)"
                    },
                    "bottleneck_size": {
                        "name": "bottleneck_size",
                        "label": "Tamanho do Gargalo",
                        "type": "slider",
                        "default": 128,
                        "min": 32,
                        "max": 512,
                        "step": 32,
                        "description": "Dimensão do gargalo na sequência 1D"
                    }
                }
            ),

            "fft": AlgorithmConfig(
                type=AlgorithmType.FFT,
                name="FFT Compression",
                description="Compressão baseada em Transformada Rápida de Fourier",
                parameters={
                    "frequency_cutoff": {
                        "name": "frequency_cutoff",
                        "label": "Corte de Frequência",
                        "type": "slider",
                        "default": 0.3,
                        "min": 0.1,
                        "max": 0.8,
                        "step": 0.05,
                        "description": "Porcentagem de frequências mantidas"
                    }
                }
            ),

            "wavelet": AlgorithmConfig(
                type=AlgorithmType.WAVELET,
                name="Wavelet Transform",
                description="Compressão multiresolução usando wavelets",
                parameters={
                    "wavelet_type": {
                        "name": "wavelet_type",
                        "label": "Tipo de Wavelet",
                        "type": "select",
                        "default": "haar",
                        "options": [
                            {"value": "haar", "label": "Haar"},
                            {"value": "db4", "label": "Daubechies 4"},
                            {"value": "sym4", "label": "Symlet 4"}
                        ],
                        "description": "Família de wavelet utilizada"
                    },
                    "threshold": {
                        "name": "threshold",
                        "label": "Threshold",
                        "type": "slider",
                        "default": 0.1,
                        "min": 0.01,
                        "max": 0.5,
                        "step": 0.01,
                        "description": "Threshold para thresholding soft"
                    }
                }
            ),

            "pca": AlgorithmConfig(
                type=AlgorithmType.PCA,
                name="PCA Compression",
                description="Compressão via Análise de Componentes Principais",
                parameters={
                    "components": {
                        "name": "components",
                        "label": "Componentes",
                        "type": "slider",
                        "default": 20,
                        "min": 5,
                        "max": 100,
                        "step": 5,
                        "description": "Número de componentes principais"
                    }
                }
            ),

            "ica": AlgorithmConfig(
                type=AlgorithmType.ICA,
                name="ICA Compression",
                description="Compressão via Análise de Componentes Independentes",
                parameters={
                    "components": {
                        "name": "components",
                        "label": "Componentes",
                        "type": "slider",
                        "default": 15,
                        "min": 5,
                        "max": 50,
                        "step": 5,
                        "description": "Número de componentes independentes"
                    },
                    "max_iter": {
                        "name": "max_iter",
                        "label": "Iterações Máximas",
                        "type": "number",
                        "default": 1000,
                        "min": 100,
                        "max": 5000,
                        "description": "Máximo de iterações para convergência"
                    }
                }
            )
        }

    def get_available_algorithms(self) -> Dict[str, AlgorithmConfig]:
        """Retornar todos os algoritmos disponíveis"""
        return self.algorithms

    def get_algorithm_info(self, algorithm_type: AlgorithmType) -> Optional[AlgorithmConfig]:
        """Obter informações de um algoritmo específico"""
        return self.algorithms.get(algorithm_type.value)

    def get_algorithm_categories(self) -> Dict[str, Dict[str, Any]]:
        """Obter categorias de algoritmos"""
        return {
            "decomposition": {
                "name": "Decomposições Matriciais",
                "description": "Métodos baseados em decomposição de matrizes",
                "algorithms": ["svd", "pca", "ica"]
            },
            "compression": {
                "name": "Compressão Clássica",
                "description": "Técnicas tradicionais de compressão",
                "algorithms": ["fft", "wavelet", "compressed-sensing"]
            },
            "neural": {
                "name": "Redes Neurais",
                "description": "Métodos baseados em aprendizado profundo",
                "algorithms": ["autoencoder", "gan"]
            },
            "hybrid": {
                "name": "Métodos Híbridos",
                "description": "Combinação de múltiplas abordagens",
                "algorithms": ["hybrid", "hilbert"]
            }
        }

    def validate_parameters(self, algorithm_type: AlgorithmType, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validar parâmetros de um algoritmo"""
        algorithm = self.get_algorithm_info(algorithm_type)
        if not algorithm:
            return False, ["Algoritmo não encontrado"]

        errors = []

        for param_name, param_config in algorithm.parameters.items():
            if param_name in parameters:
                value = parameters[param_name]

                # Validações básicas por tipo
                if param_config["type"] == "slider":
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 100)
                    if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                        errors.append(f"{param_config['label']}: deve estar entre {min_val} e {max_val}")

                elif param_config["type"] == "number":
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 1000)
                    if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                        errors.append(f"{param_config['label']}: deve estar entre {min_val} e {max_val}")

                elif param_config["type"] == "select":
                    options = param_config.get("options", [])
                    valid_values = [opt["value"] for opt in options]
                    if value not in valid_values:
                        errors.append(f"{param_config['label']}: valor inválido")

        return len(errors) == 0, errors

    def get_algorithm_defaults(self, algorithm_type: AlgorithmType) -> Optional[Dict[str, Any]]:
        """Obter parâmetros padrão de um algoritmo"""
        algorithm = self.get_algorithm_info(algorithm_type)
        if not algorithm:
            return None

        defaults = {}
        for param_name, param_config in algorithm.parameters.items():
            defaults[param_name] = param_config["default"]

        return defaults
