"""
Serviço para manipulação de imagens
"""

import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from PIL import Image
import aiofiles

from core.config import settings
from core.logging import logger


def build_full_url(path: str) -> str:
    """Construir URL completa para arquivos estáticos"""
    base_url = os.getenv("BASE_URL", "http://localhost:8001")
    return f"{base_url}{path}"

class ImageService:
    """Serviço para operações com imagens"""

    def __init__(self):
        self.metadata_file = settings.BASE_DIR / "image_metadata.json"
        self._ensure_metadata_file()

    def _ensure_metadata_file(self):
        """Garantir que o arquivo de metadados existe"""
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)

    async def _load_metadata(self) -> Dict[str, Any]:
        """Carregar metadados das imagens"""
        try:
            async with aiofiles.open(self.metadata_file, 'r') as f:
                content = await f.read()
                return json.loads(content) if content else {}
        except Exception as e:
            logger.error(f"Erro ao carregar metadados: {e}")
            return {}

    async def _save_metadata(self, metadata: Dict[str, Any]):
        """Salvar metadados das imagens"""
        try:
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))
        except Exception as e:
            logger.error(f"Erro ao salvar metadados: {e}")

    async def process_upload(
        self,
        file_content: bytes,
        filename: str,
        file_ext: str
    ) -> Dict[str, Any]:
        """Processar upload de imagem"""

        # Gerar ID único
        image_id = str(uuid.uuid4())

        # Salvar arquivo
        file_path = settings.UPLOAD_DIR / f"{image_id}{file_ext}"

        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)

        # Processar imagem com PIL
        try:
            image = Image.open(file_path)

            # Converter para RGB se necessário
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Obter dimensões
            width, height = image.size

            # Salvar como JPEG para padronização
            jpeg_path = settings.UPLOAD_DIR / f"{image_id}.jpg"
            image.save(jpeg_path, 'JPEG', quality=95)

            # Remover arquivo original se for diferente
            if file_ext != '.jpg':
                os.remove(file_path)
                file_path = jpeg_path

        except Exception as e:
            # Se der erro no processamento, manter arquivo original
            logger.warning(f"Erro ao processar imagem {filename}: {e}")
            width, height = 0, 0

        # Criar metadados
        metadata = {
            "id": image_id,
            "filename": filename,
            "original_filename": filename,
            "path": str(file_path.relative_to(settings.BASE_DIR)),
            "url": build_full_url(f"/uploads/{image_id}.jpg"),
            "size": len(file_content),
            "width": width,
            "height": height,
            "uploaded_at": datetime.now().isoformat(),
            "file_format": file_ext
        }

        # Salvar metadados
        all_metadata = await self._load_metadata()
        all_metadata[image_id] = metadata
        await self._save_metadata(all_metadata)

        return metadata

    async def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Obter informações de uma imagem"""
        metadata = await self._load_metadata()
        return metadata.get(image_id)

    async def list_images(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Listar imagens"""
        metadata = await self._load_metadata()

        # Ordenar por data de upload (mais recente primeiro)
        images = list(metadata.values())
        images.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)

        # Aplicar paginação
        start = offset
        end = offset + limit

        return images[start:end]

    async def delete_image(self, image_id: str) -> bool:
        """Deletar uma imagem"""
        metadata = await self._load_metadata()

        if image_id not in metadata:
            return False

        # Remover arquivo físico
        image_info = metadata[image_id]
        file_path = settings.BASE_DIR / image_info['path']

        try:
            if file_path.exists():
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Erro ao remover arquivo {file_path}: {e}")

        # Remover metadados
        del metadata[image_id]
        await self._save_metadata(metadata)

        return True

    async def image_exists(self, image_id: str) -> bool:
        """Verificar se uma imagem existe"""
        metadata = await self._load_metadata()
        return image_id in metadata

    async def get_image_path(self, image_id: str) -> Optional[Path]:
        """Obter caminho físico de uma imagem"""
        metadata = await self._load_metadata()
        image_info = metadata.get(image_id)

        if image_info:
            return settings.BASE_DIR / image_info['path']

        return None
