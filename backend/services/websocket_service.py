"""
Serviço para gerenciamento de conexões WebSocket
"""

from typing import Dict, List
from fastapi import WebSocket
from models.schemas import ProcessingProgress, AlgorithmType, ProcessingStatus
import json

from core.logging import logger

class WebSocketManager:
    """Gerenciador de conexões WebSocket para processamento em tempo real"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        """Conectar um WebSocket a um job específico"""
        await websocket.accept()

        if job_id not in self.active_connections:
            self.active_connections[job_id] = []

        self.active_connections[job_id].append(websocket)

        logger.info(f"WebSocket conectado para job {job_id}")

    def disconnect(self, websocket: WebSocket, job_id: str):
        """Desconectar um WebSocket"""
        if job_id in self.active_connections:
            try:
                self.active_connections[job_id].remove(websocket)
                logger.info(f"WebSocket desconectado de job {job_id}")

                # Remover lista se vazia
                if not self.active_connections[job_id]:
                    del self.active_connections[job_id]

            except ValueError:
                pass  # WebSocket já foi removido

    async def broadcast_progress(
        self,
        job_id: str,
        image_id: str,
        algorithm: AlgorithmType,
        progress: float,
        status: ProcessingStatus,
        message: str = "",
        eta: float = None
    ):
        """Enviar progresso para todos os WebSockets conectados a um job"""

        if job_id not in self.active_connections:
            return

        progress_data = ProcessingProgress(
            job_id=job_id,
            image_id=image_id,
            algorithm=algorithm,
            progress=progress,
            status=status,
            message=message,
            eta=eta
        )

        # Converter para dict para serialização JSON
        progress_dict = {
            "job_id": progress_data.job_id,
            "image_id": progress_data.image_id,
            "algorithm": progress_data.algorithm.value,
            "progress": progress_data.progress,
            "status": progress_data.status.value,
            "message": progress_data.message,
            "eta": progress_data.eta,
            "timestamp": progress_data.__class__.__name__  # Adicionar timestamp
        }

        # Enviar para todas as conexões do job
        disconnected = []
        for websocket in self.active_connections[job_id]:
            try:
                await websocket.send_json(progress_dict)
            except Exception as e:
                logger.error(f"Erro ao enviar progresso via WebSocket: {e}")
                disconnected.append(websocket)

        # Remover conexões desconectadas
        for websocket in disconnected:
            self.disconnect(websocket, job_id)

    async def broadcast_to_job(self, job_id: str, data: Dict):
        """Enviar dados arbitrários para um job específico"""

        if job_id not in self.active_connections:
            return

        disconnected = []
        for websocket in self.active_connections[job_id]:
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Erro ao enviar dados via WebSocket: {e}")
                disconnected.append(websocket)

        # Remover conexões desconectadas
        for websocket in disconnected:
            self.disconnect(websocket, job_id)

    def get_active_connections_count(self, job_id: str = None) -> int:
        """Obter número de conexões ativas"""
        if job_id:
            return len(self.active_connections.get(job_id, []))
        else:
            return sum(len(connections) for connections in self.active_connections.values())

    async def cleanup_job(self, job_id: str):
        """Limpar todas as conexões de um job"""
        if job_id in self.active_connections:
            # Fechar todas as conexões
            for websocket in self.active_connections[job_id]:
                try:
                    await websocket.close()
                except Exception:
                    pass

            del self.active_connections[job_id]
            logger.info(f"Limpando conexões WebSocket para job {job_id}")

    async def broadcast_job_completion(self, job_id: str, results: List[Dict]):
        """Notificar conclusão de um job"""
        await self.broadcast_to_job(job_id, {
            "type": "job_completed",
            "job_id": job_id,
            "results": results,
            "timestamp": str(datetime.now())
        })

    async def broadcast_job_error(self, job_id: str, error: str):
        """Notificar erro em um job"""
        await self.broadcast_to_job(job_id, {
            "type": "job_error",
            "job_id": job_id,
            "error": error,
            "timestamp": str(datetime.now())
        })
