# Compressive Sensing com Refinamento IA

Sistema completo de compressão de imagens usando Compressive Sensing (CS) L1 + Refinamento com Redes Neurais.

## Fluxo Completo

```
Imagem Original → CS L1 → Imagem Comprimida → IA Refiner → Imagem Final
```

## Uso Rápido (Tudo Automático)

### Pipeline Completo

```bash
python run_full_pipeline.py
```

Isso executa automaticamente:
1. ✅ Baixa dataset (Set5 ou Set14)
2. ✅ Gera imagens CS L1 para todas
3. ✅ Treina modelo IA
4. ✅ Testa em imagens de exemplo
5. ✅ Gera modelo pronto para uso

**Tempo estimado**: 30-60 minutos (dependendo do dataset)

### Usar Modelo Treinado

```bash
# Pipeline completo em uma imagem
python cs_pipeline.py imagem.jpg resultado.jpg

# Ou apenas refinamento IA
python cs_apply_model.py cs_refiner_model.pth img_cs.jpg resultado.jpg
```

## Estrutura de Arquivos

```
CompressedSensing/
├── run_full_pipeline.py      # ⭐ Script principal (faz tudo)
├── cs_pipeline.py            # Pipeline: Original → CS → IA
├── cs_train_dataset.py        # Treina modelo em dataset
├── cs_apply_model.py          # Aplica modelo em nova imagem
├── download_dataset_simple.py # Baixa Set5
├── download_dataset.py        # Baixa Set14/BSDS300
├── generate_cs_images.py      # Gera CS L1 em batch
├── compressive_sensing_l1.jl  # Implementação CS L1 (Julia)
└── cs_refiner_model.pth       # Modelo treinado (gerado)
```

## Passo a Passo Manual

### 1. Preparar Dataset

```bash
# Opção A: Baixar dataset pronto
python download_dataset_simple.py  # Set5 (5 imagens)

# Opção B: Usar suas imagens
python prepare_dataset.py ./minhas_imagens
```

### 2. Gerar Imagens CS L1

```bash
# Automático (processa todas)
python generate_cs_images.py

# Ou manualmente em Julia
julia compressive_sensing_l1.jl
```

### 3. Treinar Modelo

```bash
python cs_train_dataset.py
```

### 4. Usar em Nova Imagem

```bash
# Pipeline completo
python cs_pipeline.py img.jpg resultado.jpg

# Apenas refinamento
python cs_apply_model.py cs_refiner_model.pth img_cs.jpg resultado.jpg
```

## Requisitos

### Python
```bash
pip install torch torchvision pillow tqdm matplotlib numpy
```

### Julia
```bash
# Instale Julia: https://julialang.org/downloads/
# Depois instale pacotes:
julia -e 'using Pkg; Pkg.add(["Images", "LinearAlgebra", "JuMP", "GLPK"])'
```

## Resultados Esperados

| Etapa | MSE | Compressão | Tempo |
|-------|-----|------------|-------|
| Original | 0 | 100% | - |
| CS L1 (50%) | ~0.00015 | 50% | ~30s/img |
| CS L1 + IA | ~0.00008 | 50% | ~0.1s/img |

## Parâmetros

### Taxa de Amostragem CS
- `0.3` = Alta compressão (30% dados)
- `0.5` = Padrão (50% dados) ⭐
- `0.7` = Baixa compressão (70% dados)

### Treinamento
- **Épocas**: 30-50 para testes, 100+ para produção
- **Batch size**: 4-8 (depende da RAM)
- **Dataset mínimo**: 10-20 imagens

## Troubleshooting

### Julia não encontrado
```bash
# Adicione ao PATH ou use caminho completo
export PATH=$PATH:/caminho/para/julia/bin
```

### Erro ao gerar CS L1
- Verifique se Julia está instalado
- Verifique se pacotes Julia estão instalados
- CS L1 é lento (~30s por imagem)

### Modelo não melhora
- Aumente número de épocas
- Use mais imagens no dataset
- Ajuste learning rate

## Exemplos de Uso

```bash
# 1. Pipeline completo automático
python run_full_pipeline.py

# 2. Apenas treinar (se já tem dataset)
python cs_train_dataset.py

# 3. Aplicar em imagem específica
python cs_pipeline.py foto.jpg foto_refinada.jpg 0.5

# 4. Testar modelo
python cs_apply_model.py cs_refiner_model.pth teste_cs.jpg teste_final.jpg
```

## Próximos Passos

- [ ] Testar em diferentes tipos de imagens
- [ ] Ajustar hiperparâmetros
- [ ] Comparar com outros métodos
- [ ] Documentar resultados no relatório




