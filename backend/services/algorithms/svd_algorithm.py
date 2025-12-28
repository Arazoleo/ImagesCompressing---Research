"""
Implementação do algoritmo SVD para compressão de imagens
"""

import os
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple
import time

from core.config import settings
from core.logging import logger
from services.image_service import build_full_url

class SVDAlgorithm:
    """Algoritmo de compressão SVD"""

    def __init__(self):
        self.name = "SVD Compression"
        self.description = "Decomposição em Valores Singulares para compressão matricial"

    def process_image(self, image_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa uma imagem usando SVD

        Args:
            image_path: Caminho para a imagem de entrada
            parameters: Parâmetros do algoritmo (k - número de valores singulares)

        Returns:
            Dicionário com resultados do processamento
        """
        start_time = time.time()

        try:
            # Carregar imagem
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

            original_shape = image.shape
            logger.info(f"Processando imagem SVD: {original_shape}")

            # Converter para float32 e normalizar
            image_float = image.astype(np.float32) / 255.0

            # Aplicar SVD
            U, s, Vt = np.linalg.svd(image_float, full_matrices=False)

            # Parâmetro k (número de valores singulares)
            k = min(parameters.get('k', 50), len(s))
            logger.info(f"Usando k={k} valores singulares de {len(s)} disponíveis")

            # Truncar SVD
            U_k = U[:, :k]
            s_k = s[:k]
            Vt_k = Vt[:k, :]

            # Reconstruir imagem
            reconstructed = U_k @ np.diag(s_k) @ Vt_k

            # Converter de volta para imagem
            reconstructed_image = (reconstructed * 255).astype(np.uint8)
            reconstructed_image = np.clip(reconstructed_image, 0, 255)

            # Calcular métricas
            psnr = self._calculate_psnr(image, reconstructed_image)
            ssim = self._calculate_ssim(image, reconstructed_image)
            mse = self._calculate_mse(image, reconstructed_image)

            # Calcular taxa de compressão
            original_size = os.path.getsize(image_path)
            # Estimativa aproximada do tamanho comprimido
            compressed_size = (U_k.size + s_k.size + Vt_k.size) * 4  # 4 bytes por float32
            compression_ratio = original_size / max(compressed_size, 1)

            # Salvar imagem processada
            output_filename = f"svd_{Path(image_path).stem}_k{k}.jpg"
            output_path = settings.PROCESSED_DIR / output_filename
            cv2.imwrite(str(output_path), reconstructed_image)

            processing_time = time.time() - start_time

            logger.info(f"SVD processado com sucesso: PSNR={psnr:.2f}, SSIM={ssim:.3f}, Ratio={compression_ratio:.2f}")

            return {
                "success": True,
                "algorithm": "svd",
                "output_path": str(output_path),
                "output_url": build_full_url(f"/processed/{output_filename}"),
                "metrics": {
                    "psnr": round(psnr, 2),
                    "ssim": round(ssim, 3),
                    "mse": round(mse, 4),
                    "compression_ratio": round(compression_ratio, 2),
                    "file_size": output_path.stat().st_size
                },
                "parameters_used": {
                    "k": k,
                    "total_singular_values": len(s)
                },
                "processing_time": round(processing_time, 2)
            }

        except Exception as e:
            logger.error(f"Erro no processamento SVD: {str(e)}")
            return {
                "success": False,
                "algorithm": "svd",
                "error": str(e),
                "processing_time": round(time.time() - start_time, 2)
            }

    def _calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calcula PSNR entre duas imagens"""
        mse = self._calculate_mse(original, reconstructed)
        if mse == 0:
            return float('inf')

        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def _calculate_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calcula MSE entre duas imagens"""
        return np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)

    def _calculate_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calcula SSIM entre duas imagens usando OpenCV"""
        try:
            # SSIM do OpenCV
            ssim_score = cv2.quality.QualitySSIM_compute(
                cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if len(original.shape) == 2 else original,
                cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2BGR) if len(reconstructed.shape) == 2 else reconstructed
            )[0]
            return ssim_score
        except:
            # Fallback: SSIM simples
            return self._simple_ssim(original, reconstructed)

    def _simple_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Implementação simples de SSIM"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1*img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator
        return np.mean(ssim_map)
