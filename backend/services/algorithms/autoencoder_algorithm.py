import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import time
from typing import Dict, Any
import os

from core.config import settings
from services.image_service import build_full_url


class Autoencoder(nn.Module):
    """Autoencoder convolucional para compressão de imagens"""

    def __init__(self, latent_dim: int = 128):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 256x256 -> 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (128, 32, 32)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),    # 128x128 -> 256x256
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderAlgorithm:
    """Algoritmo de compressão usando Autoencoder"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.latent_dim = 128

    def _load_or_create_model(self, latent_dim: int = 128):
        """Carrega modelo treinado ou cria um novo"""
        self.latent_dim = latent_dim
        model_path = settings.PROCESSED_DIR / f"autoencoder_latent_{latent_dim}.pth"

        self.model = Autoencoder(latent_dim).to(self.device)

        if model_path.exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Modelo carregado de {model_path}")
            except Exception as e:
                print(f"Erro ao carregar modelo: {e}. Usando modelo não treinado.")
        else:
            print("Modelo não encontrado. Usando modelo não treinado (para teste).")
            # Por enquanto, vamos usar o modelo sem treinamento para testar

        self.model.eval()

    def _train_model(self):
        """Treina o autoencoder com dados sintéticos"""
        print("Treinando autoencoder...")

        # Criar dados de treinamento sintéticos (imagens simples)
        train_data = self._generate_training_data(50)  # Reduzido para 50
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)  # Batch menor

        # Configurar otimizador e loss
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Treinar por menos épocas
        self.model.train()
        for epoch in range(5):  # Reduzido para 5 épocas
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                output = self.model(batch)
                loss = criterion(output, batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(".4f")

        # Salvar modelo
        model_path = settings.PROCESSED_DIR / f"autoencoder_latent_{self.latent_dim}.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Modelo salvo em {model_path}")

    def _generate_training_data(self, num_samples: int = 100):
        """Gera dados de treinamento sintéticos"""
        images = []

        for _ in range(num_samples):
            # Criar imagem simples com padrões
            img = np.zeros((256, 256), dtype=np.float32)

            # Adicionar alguns círculos e retângulos aleatórios
            for _ in range(5):
                if np.random.random() > 0.5:
                    # Círculo
                    center = (np.random.randint(50, 200), np.random.randint(50, 200))
                    radius = np.random.randint(10, 30)
                    cv2.circle(img, center, radius, 1.0, -1)
                else:
                    # Retângulo
                    pt1 = (np.random.randint(0, 200), np.random.randint(0, 200))
                    pt2 = (pt1[0] + np.random.randint(20, 50), pt1[1] + np.random.randint(20, 50))
                    cv2.rectangle(img, pt1, pt2, 1.0, -1)

            images.append(img)

        return torch.tensor(np.array(images)).unsqueeze(1)  # Adicionar dimensão do canal

    def _calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calcula PSNR entre imagem original e reconstruída"""
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def _calculate_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calcula SSIM aproximado"""
        # SSIM simplificado (versão básica)
        mu1 = np.mean(original)
        mu2 = np.mean(reconstructed)
        sigma1_sq = np.var(original)
        sigma2_sq = np.var(reconstructed)
        sigma12 = np.cov(original.flatten(), reconstructed.flatten())[0, 1]

        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2

        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

        return numerator / denominator

    def process_image(self, image_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()

        try:
            # Carregar imagem usando PIL (sem OpenCV)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = image.resize((256, 256), Image.Resampling.LANCZOS)

            # Converter para numpy array e normalizar
            image_array = np.array(image, dtype=np.float32) / 255.0

            # Carregar ou treinar modelo
            latent_dim = parameters.get('latent_dim', 128)
            self._load_or_create_model(latent_dim)

            # Preparar tensor
            image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0).to(self.device)

            # Processar com autoencoder
            with torch.no_grad():
                reconstructed_tensor = self.model(image_tensor)
                reconstructed = reconstructed_tensor.squeeze().cpu().numpy()

            # Garantir valores entre 0 e 1
            reconstructed = np.clip(reconstructed, 0, 1)

            # Converter para uint8
            reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)

            # Salvar imagem processada usando PIL
            output_filename = f"autoencoder_{Path(image_path).stem}_latent{latent_dim}.png"
            output_path = settings.PROCESSED_DIR / output_filename
            reconstructed_image = Image.fromarray(reconstructed_uint8, mode='L')
            reconstructed_image.save(str(output_path))

            # Calcular métricas
            psnr = self._calculate_psnr(image_array, reconstructed)
            ssim = self._calculate_ssim(image_array, reconstructed)

            # Calcular ratio de compressão aproximado
            original_size = os.path.getsize(image_path)
            compressed_size = output_path.stat().st_size
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            return {
                "success": True,
                "algorithm": "autoencoder",
                "output_path": str(output_path),
                "output_url": build_full_url(f"/processed/{output_filename}"),
                "metrics": {
                    "psnr": round(psnr, 2),
                    "ssim": round(ssim, 3),
                    "compression_ratio": round(compression_ratio, 2),
                    "file_size": output_path.stat().st_size
                },
                "parameters_used": {
                    "latent_dim": latent_dim
                },
                "processing_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            return {
                "success": False,
                "algorithm": "autoencoder",
                "error": str(e),
                "processing_time": round(time.time() - start_time, 2)
            }
