"""
Modelos Pydantic para requisições e respostas da API
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class AlgorithmType(str, Enum):
    """Tipos de algoritmos suportados"""
    SVD = "svd"
    COMPRESSED_SENSING = "compressed-sensing"
    AUTOENCODER = "autoencoder"
    HYBRID = "hybrid"
    HILBERT = "hilbert"
    FFT = "fft"
    WAVELET = "wavelet"
    PCA = "pca"
    ICA = "ica"
    GAN = "gan"

class ProcessingStatus(str, Enum):
    """Status do processamento"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

# --- Upload e Imagens ---

class ImageUploadResponse(BaseModel):
    """Resposta do upload de imagem"""
    id: str = Field(..., description="ID único da imagem")
    filename: str = Field(..., description="Nome do arquivo")
    url: str = Field(..., description="URL para acessar a imagem")
    size: int = Field(..., description="Tamanho em bytes")
    width: int = Field(..., description="Largura da imagem")
    height: int = Field(..., description="Altura da imagem")
    uploaded_at: str = Field(..., description="Data/hora do upload")

class ImageInfo(BaseModel):
    """Informações básicas de uma imagem"""
    id: str
    filename: str
    url: str
    size: int
    width: int
    height: int

# --- Algoritmos e Parâmetros ---

class AlgorithmParameter(BaseModel):
    """Parâmetro de um algoritmo"""
    name: str = Field(..., description="Nome do parâmetro")
    value: Any = Field(..., description="Valor do parâmetro")

class AlgorithmConfig(BaseModel):
    """Configuração de um algoritmo para processamento"""
    type: AlgorithmType
    parameters: Dict[str, Any] = Field(default_factory=dict)

# --- Processamento ---

class ProcessingRequest(BaseModel):
    """Requisição de processamento"""
    image_ids: List[str] = Field(..., description="IDs das imagens a processar")
    algorithms: List[AlgorithmConfig] = Field(..., description="Algoritmos a aplicar")

    @validator('image_ids')
    def validate_image_ids(cls, v):
        if not v:
            raise ValueError('Pelo menos uma imagem deve ser especificada')
        if len(v) > 10:
            raise ValueError('Máximo de 10 imagens por processamento')
        return v

class ProcessingMetrics(BaseModel):
    """Métricas de qualidade da imagem processada"""
    psnr: Optional[float] = Field(None, description="Peak Signal-to-Noise Ratio (dB)")
    ssim: Optional[float] = Field(None, description="Structural Similarity Index")
    mse: Optional[float] = Field(None, description="Mean Squared Error")
    compression_ratio: Optional[float] = Field(None, description="Taxa de compressão")
    file_size: Optional[int] = Field(None, description="Tamanho do arquivo resultante")

class ProcessingResult(BaseModel):
    """Resultado de um processamento"""
    id: str = Field(..., description="ID único do resultado")
    image_id: str = Field(..., description="ID da imagem original")
    algorithm: AlgorithmType = Field(..., description="Algoritmo utilizado")
    status: ProcessingStatus = Field(..., description="Status do processamento")
    result_url: Optional[str] = Field(None, description="URL da imagem processada")
    metrics: Optional[ProcessingMetrics] = Field(None, description="Métricas de qualidade")
    processing_time: Optional[float] = Field(None, description="Tempo de processamento (segundos)")
    error_message: Optional[str] = Field(None, description="Mensagem de erro se houver")
    created_at: str = Field(..., description="Data/hora da criação")

class BatchProcessingResponse(BaseModel):
    """Resposta do processamento em lote"""
    job_id: str = Field(..., description="ID do job de processamento")
    total_images: int = Field(..., description="Total de imagens")
    total_algorithms: int = Field(..., description="Total de algoritmos")
    estimated_time: float = Field(..., description="Tempo estimado (segundos)")
    results: List[ProcessingResult] = Field(default_factory=list)
    progress: float = Field(0.0, description="Progresso do processamento (0-100)")
    status: str = Field("running", description="Status do job")

# --- Comparação ---

class ComparisonRequest(BaseModel):
    """Requisição de comparação de algoritmos"""
    image_id: str = Field(..., description="ID da imagem para comparação")
    algorithms: List[AlgorithmConfig] = Field(..., min_items=2, description="Algoritmos para comparar")

class AlgorithmComparison(BaseModel):
    """Comparação entre algoritmos"""
    algorithm: AlgorithmType
    result_url: Optional[str]
    metrics: Optional[ProcessingMetrics]
    rank: Optional[int] = Field(None, description="Ranking baseado na qualidade")

class ComparisonResponse(BaseModel):
    """Resposta da comparação"""
    id: str
    image_id: str
    comparisons: List[AlgorithmComparison]
    best_algorithm: AlgorithmType
    created_at: str

# --- WebSocket ---

class ProcessingProgress(BaseModel):
    """Progresso do processamento via WebSocket"""
    job_id: str
    image_id: str
    algorithm: AlgorithmType
    progress: float = Field(..., ge=0, le=100, description="Progresso em %")
    status: ProcessingStatus
    message: Optional[str] = None
    eta: Optional[float] = None  # Tempo estimado restante

# --- Respostas Gerais ---

class APIResponse(BaseModel):
    """Resposta genérica da API"""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None

class HealthResponse(BaseModel):
    """Resposta do health check"""
    status: str
    service: str
    version: str
    uptime: Optional[float] = None
