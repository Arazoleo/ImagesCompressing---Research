"""
Router para informações sobre algoritmos disponíveis
"""

from typing import Dict, List
from fastapi import APIRouter, HTTPException

from core.logging import logger
from models.schemas import AlgorithmType, AlgorithmConfig
from services.algorithm_service import AlgorithmService

router = APIRouter()
algorithm_service = AlgorithmService()

@router.get("/algorithms", response_model=Dict[str, AlgorithmConfig])
async def get_available_algorithms():
    """
    Listar todos os algoritmos disponíveis
    """
    try:
        algorithms = algorithm_service.get_available_algorithms()
        return algorithms

    except Exception as e:
        logger.error(f"Erro ao obter algoritmos: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.get("/algorithms/{algorithm_type}", response_model=AlgorithmConfig)
async def get_algorithm_info(algorithm_type: AlgorithmType):
    """
    Obter informações detalhadas de um algoritmo específico
    """
    try:
        algorithm = algorithm_service.get_algorithm_info(algorithm_type)
        if not algorithm:
            raise HTTPException(status_code=404, detail="Algoritmo não encontrado")

        return algorithm

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter info do algoritmo {algorithm_type}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.get("/algorithms/categories")
async def get_algorithm_categories():
    """
    Obter categorias de algoritmos disponíveis
    """
    try:
        categories = algorithm_service.get_algorithm_categories()
        return categories

    except Exception as e:
        logger.error(f"Erro ao obter categorias: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.post("/algorithms/{algorithm_type}/validate")
async def validate_algorithm_parameters(
    algorithm_type: AlgorithmType,
    parameters: Dict = None
):
    """
    Validar parâmetros de um algoritmo
    """
    try:
        if parameters is None:
            parameters = {}

        is_valid, errors = algorithm_service.validate_parameters(
            algorithm_type,
            parameters
        )

        return {
            "valid": is_valid,
            "errors": errors if not is_valid else []
        }

    except Exception as e:
        logger.error(f"Erro ao validar parâmetros do algoritmo {algorithm_type}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.get("/algorithms/{algorithm_type}/defaults")
async def get_algorithm_defaults(algorithm_type: AlgorithmType):
    """
    Obter parâmetros padrão de um algoritmo
    """
    try:
        defaults = algorithm_service.get_algorithm_defaults(algorithm_type)
        if defaults is None:
            raise HTTPException(status_code=404, detail="Algoritmo não encontrado")

        return defaults

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter defaults do algoritmo {algorithm_type}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")
