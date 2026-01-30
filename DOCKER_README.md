# 🐳 ImageStudio - Docker Setup

Este guia mostra como executar o ImageStudio usando Docker e Docker Compose.

## 📋 Pré-requisitos

- Docker instalado (versão 20.10+)
- Docker Compose instalado (versão 2.0+)

## 🚀 Início Rápido

### 1. Clone o repositório (se necessário)
```bash
git clone <repository-url>
cd ImagesCompressing---Research
```

### 2. Execute com Docker Compose
```bash
# Construir e executar todos os serviços
docker-compose up --build

# Ou em background
docker-compose up -d --build
```

### 3. Acesse a aplicação
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8001
- **Documentação API:** http://localhost:8001/docs

## 🛠️ Serviços Disponíveis

### Backend (FastAPI)
- **Porta:** 8001
- **Tecnologias:** Python 3.9, FastAPI, OpenCV, NumPy
- **Funcionalidades:** Processamento de imagens, algoritmos de compressão

### Frontend (Next.js)
- **Porta:** 3000
- **Tecnologias:** Next.js, TypeScript, Tailwind CSS
- **Funcionalidades:** Interface web elegante, visualização de imagens

### Redis (Cache - Opcional)
- **Porta:** 6379
- **Uso:** Cache de resultados e sessões

## 📁 Estrutura de Volumes

```
backend/
├── uploads/      # Imagens carregadas
├── processed/    # Imagens processadas
├── temp/         # Arquivos temporários
└── logs/         # Logs da aplicação
```

## 🔧 Comandos Úteis

```bash
# Parar todos os serviços
docker-compose down

# Ver logs
docker-compose logs -f

# Ver logs de um serviço específico
docker-compose logs -f backend

# Reconstruir e executar
docker-compose up --build --force-recreate

# Limpar tudo (containers, volumes, imagens)
docker-compose down -v --rmi all
```

## 🐛 Troubleshooting

### Problema: Porta já em uso
```bash
# Verificar processos usando portas
lsof -i :3000
lsof -i :8001

# Matar processos
kill -9 <PID>
```

### Problema: Erro de build
```bash
# Limpar cache do Docker
docker system prune -a

# Reconstruir sem cache
docker-compose build --no-cache
```

### Problema: Volumes não funcionam
```bash
# Verificar permissões
ls -la backend/uploads/

# Ajustar permissões
chmod -R 755 backend/uploads/
```

## 🔒 Segurança

### Para produção, considere:
- Configurar HTTPS (nginx reverse proxy)
- Usar secrets para variáveis sensíveis
- Configurar firewall
- Usar imagens oficiais e atualizadas
- Implementar rate limiting

## 📊 Monitoramento

### Logs
```bash
# Ver logs em tempo real
docker-compose logs -f

# Ver logs de erro apenas
docker-compose logs | grep ERROR
```

### Recursos
```bash
# Ver uso de recursos
docker stats

# Ver containers ativos
docker ps
```

## 🚀 Deploy em Produção

### Usando Docker Compose
```bash
# Arquivo de produção
docker-compose -f docker-compose.prod.yml up -d
```

### Usando Docker Swarm
```bash
# Inicializar swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml imagestudio
```

## 📞 Suporte

Para problemas específicos:
1. Verifique os logs: `docker-compose logs`
2. Teste a conectividade: `curl http://localhost:8001/health`
3. Verifique portas: `docker ps`

---

**🎉 Pronto! Seu ImageStudio está rodando com Docker!**
