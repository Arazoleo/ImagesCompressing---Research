"""
Router para operações com imagens (upload, listagem, etc.)
"""

import os
import uuid
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from PIL import Image

from core.config import settings
from core.logging import logger
from models.schemas import ImageUploadResponse, ImageInfo, APIResponse
from services.image_service import ImageService

router = APIRouter()
image_service = ImageService()

@router.post("/images/upload", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload de uma imagem para processamento
    """
    try:
        # Validar tipo de arquivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nome do arquivo não fornecido")

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de arquivo não suportado. Use: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        # Validar tamanho do arquivo
        file_content = await file.read()
        if len(file_content) > settings.MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Arquivo muito grande. Máximo: {settings.MAX_IMAGE_SIZE / (1024*1024):.1f}MB"
            )

        # Processar imagem
        image_info = await image_service.process_upload(
            file_content,
            file.filename,
            file_ext
        )

        logger.info(f"Imagem carregada: {image_info['id']} - {file.filename}")

        return ImageUploadResponse(**image_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.post("/images/upload-batch", response_model=List[ImageUploadResponse])
async def upload_images_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload em lote de múltiplas imagens
    """
    try:
        if len(files) > settings.MAX_IMAGES_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Máximo de {settings.MAX_IMAGES_PER_REQUEST} imagens por vez"
            )

        results = []
        for file in files:
            try:
                # Validar tipo de arquivo
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in settings.ALLOWED_EXTENSIONS:
                    logger.warning(f"Tipo de arquivo não suportado: {file.filename}")
                    continue

                # Validar tamanho
                file_content = await file.read()
                if len(file_content) > settings.MAX_IMAGE_SIZE:
                    logger.warning(f"Arquivo muito grande: {file.filename}")
                    continue

                # Processar imagem
                image_info = await image_service.process_upload(
                    file_content,
                    file.filename,
                    file_ext
                )

                results.append(ImageUploadResponse(**image_info))

            except Exception as e:
                logger.error(f"Erro ao processar {file.filename}: {str(e)}")
                continue

        if not results:
            raise HTTPException(status_code=400, detail="Nenhuma imagem válida foi carregada")

        logger.info(f"Batch upload concluído: {len(results)} imagens")
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no batch upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/images", response_model=List[ImageInfo])
async def list_images(limit: int = 50, offset: int = 0):
    """
    Listar imagens carregadas
    """
    try:
        images = await image_service.list_images(limit=limit, offset=offset)
        return [ImageInfo(**img) for img in images]

    except Exception as e:
        logger.error(f"Erro ao listar imagens: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao listar imagens")

@router.get("/images/{image_id}")
async def get_image(image_id: str):
    """
    Obter informações de uma imagem específica
    """
    try:
        image_info = await image_service.get_image_info(image_id)
        if not image_info:
            raise HTTPException(status_code=404, detail="Imagem não encontrada")

        return ImageInfo(**image_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter imagem {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.delete("/images/{image_id}", response_model=APIResponse)
async def delete_image(image_id: str):
    """
    Deletar uma imagem
    """
    try:
        success = await image_service.delete_image(image_id)
        if not success:
            raise HTTPException(status_code=404, detail="Imagem não encontrada")

        logger.info(f"Imagem deletada: {image_id}")
        return APIResponse(
            success=True,
            message="Imagem deletada com sucesso"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao deletar imagem {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao deletar imagem")

@router.get("/images/{image_id}/download")
async def download_image(image_id: str):
    """
    Download de uma imagem processada
    """
    try:
        image_path = settings.PROCESSED_DIR / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = settings.UPLOAD_DIR / f"{image_id}.jpg"
            if not image_path.exists():
                raise HTTPException(status_code=404, detail="Imagem não encontrada")

        return FileResponse(
            path=image_path,
            filename=f"processed_{image_id}.jpg",
            media_type="image/jpeg"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no download da imagem {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro no download")
