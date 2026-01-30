"""
ImageStudio API - Backend para processamento de imagens
Combina frontend Next.js com algoritmos de álgebra linear e IA
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

from api.routes import images, processing, algorithms
from core.config import settings
from core.logging import setup_logging

# Setup logging
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciamento do ciclo de vida da aplicação"""
    # Startup
    print("🚀 Starting ImageStudio API...")

    # Criar diretórios necessários
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
    os.makedirs(settings.TEMP_DIR, exist_ok=True)

    yield

    # Shutdown
    print("🛑 Shutting down ImageStudio API...")

# Criar aplicação FastAPI
app = FastAPI(
    title="ImageStudio API",
    description="API para processamento de imagens usando álgebra linear e IA",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS para conectar com frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://frontend:3000",
        "http://frontend-1:3000",
        "http://imagescompressing---research-frontend-1:3000",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar diretórios estáticos
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/processed", StaticFiles(directory=settings.PROCESSED_DIR), name="processed")

# Incluir routers
app.include_router(images.router, prefix="/api/v1", tags=["images"])
app.include_router(processing.router, prefix="/api/v1", tags=["processing"])
app.include_router(algorithms.router, prefix="/api/v1", tags=["algorithms"])

@app.get("/")
async def root():
    """Endpoint raiz da API"""
    return {
        "message": "🎨 ImageStudio API - Processamento de Imagens",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    """Verificação de saúde da API"""
    return {
        "status": "healthy",
        "service": "ImageStudio API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
