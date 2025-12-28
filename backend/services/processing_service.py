"""
Serviço para processamento de imagens com algoritmos
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import json

from core.config import settings
from core.logging import logger
from models.schemas import (
    AlgorithmType,
    ProcessingResult,
    ProcessingStatus,
    ProcessingMetrics,
    BatchProcessingResponse
)
from services.image_service import ImageService
from services.algorithm_service import AlgorithmService
from services.algorithm_registry import algorithm_registry
from services.websocket_service import WebSocketManager

class ProcessingService:
    """Serviço para processamento de imagens"""

    def __init__(self):
        self.image_service = ImageService()
        self.algorithm_service = AlgorithmService()
        self.websocket_manager = WebSocketManager()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=settings.CONCURRENT_PROCESSES)

    async def start_batch_processing(
        self,
        image_ids: List[str],
        algorithms: List[Dict[str, Any]]
    ) -> str:
        """Iniciar processamento em lote"""

        job_id = str(uuid.uuid4())

        # Criar job
        job = {
            "id": job_id,
            "image_ids": image_ids,
            "algorithms": algorithms,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "results": [],
            "progress": 0
        }

        self.jobs[job_id] = job

        # Iniciar processamento em background
        asyncio.create_task(self._process_batch(job_id))

        logger.info(f"Job iniciado: {job_id} - {len(image_ids)} imagens, {len(algorithms)} algoritmos")
        return job_id

    async def _process_batch(self, job_id: str):
        """Processar batch em background"""

        job = self.jobs.get(job_id)
        if not job:
            return

        try:
            total_operations = len(job["image_ids"]) * len(job["algorithms"])
            completed_operations = 0

            for image_id in job["image_ids"]:
                for algorithm_config in job["algorithms"]:
                    # algorithm_config é sempre um AlgorithmConfig object
                    algorithm_type = algorithm_config.type.value
                    parameters = algorithm_config.parameters

                    # Calcular progresso inicial
                    initial_progress = (completed_operations / total_operations) * 100
                    job["progress"] = initial_progress
                    
                    # Notificar progresso via WebSocket
                    await self.websocket_manager.broadcast_progress(
                        job_id,
                        image_id,
                        algorithm_type,
                        int(initial_progress),
                        "processing",
                        f"Iniciando {algorithm_type}..."
                    )
                    
                    logger.info(f"Iniciando {algorithm_type} para {image_id} - Progresso: {initial_progress:.1f}%")

                    try:
                        logger.info(f"Iniciando processamento: {algorithm_type} para imagem {image_id}")
                        
                        # Executar algoritmo
                        result = await self._execute_algorithm(
                            image_id,
                            algorithm_type,
                            parameters,
                            job_id
                        )

                        logger.info(f"Resultado recebido: {result.get('status', 'unknown')} para {algorithm_type}")
                        
                        job["results"].append(result)
                        completed_operations += 1

                        # Atualizar progresso
                        progress = (completed_operations / total_operations) * 100
                        job["progress"] = progress
                        
                        logger.info(f"Progresso atualizado: {completed_operations}/{total_operations} ({progress:.1f}%)")

                        # Notificar conclusão
                        await self.websocket_manager.broadcast_progress(
                            job_id,
                            image_id,
                            algorithm_type,
                            100,
                            "completed",
                            f"Concluído: {algorithm_type}"
                        )

                        logger.info(f"Algoritmo concluído: {algorithm_type} para imagem {image_id}")

                    except Exception as e:
                        logger.error(f"Erro no algoritmo {algorithm_type} para imagem {image_id}: {str(e)}")

                        # Notificar erro
                        await self.websocket_manager.broadcast_progress(
                            job_id,
                            image_id,
                            algorithm_type,
                            0,
                            "error",
                            f"Erro: {str(e)}"
                        )

                        # Adicionar resultado de erro
                        error_result = ProcessingResult(
                            id=f"{algorithm_type}-{image_id}",
                            image_id=image_id,
                            algorithm=algorithm_type,
                            status=ProcessingStatus.ERROR,
                            error_message=str(e),
                            created_at=datetime.now().isoformat()
                        )
                        job["results"].append(error_result.model_dump())

            # Finalizar job
            job["status"] = "completed"
            logger.info(f"Job concluído: {job_id}")

        except Exception as e:
            logger.error(f"Erro no processamento do job {job_id}: {str(e)}")
            job["status"] = "error"
            job["error"] = str(e)

    async def _execute_algorithm(
        self,
        image_id: str,
        algorithm_type: AlgorithmType,
        parameters: Dict[str, Any],
        job_id: str
    ) -> Dict[str, Any]:
        """Executar um algoritmo específico"""

        result_id = f"{algorithm_type}-{image_id}"

        try:
            # Verificar se o algoritmo está disponível
            if not algorithm_registry.is_algorithm_available(algorithm_type):
                raise ValueError(f"Algoritmo '{algorithm_type}' não implementado")

            # Obter caminho da imagem
            image_path = await self.image_service.get_image_path(image_id)
            if not image_path:
                raise ValueError(f"Imagem {image_id} não encontrada")

            logger.info(f"Executando {algorithm_type} na imagem {image_id}")

            # Obter algoritmo e executar
            algorithm = algorithm_registry.get_algorithm(algorithm_type)

            # Executar em thread separada para não bloquear o event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                algorithm.process_image,
                str(image_path),
                parameters
            )

            if result["success"]:
                # Criar resultado de sucesso
                processing_result = ProcessingResult(
                    id=result_id,
                    image_id=image_id,
                    algorithm=algorithm_type,
                    status=ProcessingStatus.COMPLETED,
                    result_url=result["output_url"],
                    metrics=ProcessingMetrics(**result["metrics"]),
                    processing_time=result["processing_time"],
                    created_at=datetime.now().isoformat()
                )
                logger.info(f"Algoritmo {algorithm_type} concluído com sucesso: PSNR={result['metrics']['psnr']:.2f}")
            else:
                # Criar resultado de erro
                processing_result = ProcessingResult(
                    id=result_id,
                    image_id=image_id,
                    algorithm=algorithm_type,
                    status=ProcessingStatus.ERROR,
                    error_message=result.get("error", "Erro desconhecido"),
                    processing_time=result.get("processing_time", 0),
                    created_at=datetime.now().isoformat()
                )
                logger.error(f"Erro no algoritmo {algorithm_type}: {result.get('error', 'Erro desconhecido')}")

            return processing_result.model_dump()

        except Exception as e:
            logger.error(f"Erro executando algoritmo {algorithm_type}: {str(e)}")

            # Retornar resultado de erro
            processing_result = ProcessingResult(
                id=result_id,
                image_id=image_id,
                algorithm=algorithm_type,
                status=ProcessingStatus.ERROR,
                error_message=str(e),
                processing_time=0,
                created_at=datetime.now().isoformat()
            )

            return processing_result.model_dump()

    async def get_job_status(self, job_id: str) -> Optional[BatchProcessingResponse]:
        """Obter status de um job"""
        job = self.jobs.get(job_id)
        if not job:
            return None

        return BatchProcessingResponse(
            job_id=job_id,
            total_images=len(job["image_ids"]),
            total_algorithms=len(job["algorithms"]),
            estimated_time=len(job["image_ids"]) * len(job["algorithms"]) * 2.5,
            results=job["results"],
            progress=job.get("progress", 0.0),
            status=job.get("status", "running")
        )

    async def get_job_results(self, job_id: str) -> Optional[List[ProcessingResult]]:
        """Obter resultados de um job"""
        job = self.jobs.get(job_id)
        if not job:
            return None

        return [ProcessingResult(**result) for result in job["results"]]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancelar um job"""
        job = self.jobs.get(job_id)
        if not job:
            return False

        job["status"] = "cancelled"
        logger.info(f"Job cancelado: {job_id}")
        return True

    async def get_processing_history(self, limit: int = 50, offset: int = 0) -> List[ProcessingResult]:
        """Obter histórico de processamentos"""
        all_results = []

        # Coletar resultados de todos os jobs
        for job in self.jobs.values():
            if job["status"] == "completed":
                all_results.extend(job["results"])

        # Ordenar por data (mais recente primeiro)
        all_results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Aplicar paginação
        start = offset
        end = offset + limit

        return [ProcessingResult(**result) for result in all_results[start:end]]

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Obter estatísticas de processamento"""
        total_jobs = len(self.jobs)
        completed_jobs = sum(1 for job in self.jobs.values() if job["status"] == "completed")
        total_images_processed = sum(len(job["image_ids"]) for job in self.jobs.values())
        total_algorithms_used = sum(len(job["algorithms"]) for job in self.jobs.values())

        # Algoritmos mais usados
        algorithm_usage = {}
        for job in self.jobs.values():
            for alg in job["algorithms"]:
                alg_type = alg.type.value
                algorithm_usage[alg_type] = algorithm_usage.get(alg_type, 0) + 1

        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "completion_rate": completed_jobs / total_jobs if total_jobs > 0 else 0,
            "total_images_processed": total_images_processed,
            "total_algorithms_used": total_algorithms_used,
            "most_used_algorithms": algorithm_usage
        }

    async def image_exists(self, image_id: str) -> bool:
        """Verificar se uma imagem existe"""
        return await self.image_service.image_exists(image_id)
