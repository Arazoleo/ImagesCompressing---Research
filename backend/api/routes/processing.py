"""
Router para processamento de imagens
"""

import asyncio
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from core.config import settings
from core.logging import logger
from models.schemas import (
    ProcessingRequest,
    BatchProcessingResponse,
    ProcessingResult,
    ProcessingProgress,
    APIResponse
)
from services.processing_service import ProcessingService
from services.websocket_service import WebSocketManager

router = APIRouter()
processing_service = ProcessingService()
websocket_manager = WebSocketManager()

@router.post("/processing", response_model=BatchProcessingResponse)
async def process_images(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """
    Processar imagens com algoritmos selecionados
    """
    try:
        # Validar imagens
        for image_id in request.image_ids:
            image_exists = await processing_service.image_exists(image_id)
            if not image_exists:
                raise HTTPException(
                    status_code=404,
                    detail=f"Imagem {image_id} não encontrada"
                )

        # Iniciar processamento em background
        job_id = await processing_service.start_batch_processing(
            request.image_ids,
            request.algorithms
        )

        # Estimar tempo
        total_operations = len(request.image_ids) * len(request.algorithms)
        estimated_time = total_operations * 2.5  # 2.5s por operação média

        # Retornar resposta imediata
        return BatchProcessingResponse(
            job_id=job_id,
            total_images=len(request.image_ids),
            total_algorithms=len(request.algorithms),
            estimated_time=estimated_time,
            results=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/processing/{job_id}/status", response_model=BatchProcessingResponse)
async def get_processing_status(job_id: str):
    """
    Obter status de um job de processamento
    """
    try:
        status = await processing_service.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job não encontrado")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter status do job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.get("/processing/{job_id}/results", response_model=List[ProcessingResult])
async def get_processing_results(job_id: str):
    """
    Obter resultados de um job de processamento
    """
    try:
        results = await processing_service.get_job_results(job_id)
        if results is None:
            raise HTTPException(status_code=404, detail="Job não encontrado")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter resultados do job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.delete("/processing/{job_id}", response_model=APIResponse)
async def cancel_processing(job_id: str):
    """
    Cancelar um job de processamento
    """
    try:
        success = await processing_service.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job não encontrado")

        logger.info(f"Job cancelado: {job_id}")
        return APIResponse(
            success=True,
            message="Processamento cancelado com sucesso"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao cancelar job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.websocket("/ws/processing/{job_id}")
async def processing_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket para acompanhar progresso em tempo real
    """
    await websocket_manager.connect(websocket, job_id)

    try:
        while True:
            # Manter conexão viva
            data = await websocket.receive_text()
            # Processar mensagens se necessário

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, job_id)
    except Exception as e:
        logger.error(f"Erro no WebSocket {job_id}: {str(e)}")
        websocket_manager.disconnect(websocket, job_id)

@router.get("/processing/history", response_model=List[ProcessingResult])
async def get_processing_history(limit: int = 50, offset: int = 0):
    """
    Obter histórico de processamentos
    """
    try:
        history = await processing_service.get_processing_history(
            limit=limit,
            offset=offset
        )
        return history

    except Exception as e:
        logger.error(f"Erro ao obter histórico: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.get("/processing/stats")
async def get_processing_stats():
    """
    Estatísticas de processamento
    """
    try:
        stats = await processing_service.get_processing_stats()
        return stats

    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")
