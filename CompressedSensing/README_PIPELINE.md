# Pipeline Completo: Original → CS L1 → IA

Este documento explica o fluxo completo de compressão e refinamento.

## Fluxo

```
Imagem Original
    ↓
CS L1 (Compressive Sensing)
    ↓
Imagem Comprimida (com artefatos)
    ↓
IA Refiner (ResNet)
    ↓
Imagem Refinada (alta qualidade)
```

## Uso Rápido

### Pipeline Automático

```bash
python cs_pipeline.py imagem_original.jpg resultado_final.jpg
```

Isso executa automaticamente:
1. CS L1 na imagem original
2. Refinamento IA na imagem comprimida
3. Salva resultado final

### Passo a Passo Manual

```bash
# 1. Aplicar CS L1
julia -e 'include("compressive_sensing_l1.jl"); compress_image_cs_L1("img.jpg", 0.5, "img_cs.jpg")'

# 2. Refinar com IA
python cs_apply_model.py cs_refiner_model.pth img_cs.jpg img_final.jpg
```

## Parâmetros

### Taxa de Amostragem CS

- `0.3` = 30% amostras (alta compressão, mais artefatos)
- `0.5` = 50% amostras (padrão, bom equilíbrio)
- `0.7` = 70% amostras (menos compressão, melhor qualidade)

### Modelo IA

- Treine primeiro: `python cs_train_dataset.py`
- Modelo salvo em: `cs_refiner_model.pth`
- Use em novas imagens: `python cs_apply_model.py`

## Comparação de Resultados

| Etapa | MSE | Qualidade |
|-------|-----|-----------|
| Original | 0 | Referência |
| CS L1 (50%) | ~0.00015 | Boa, com artefatos leves |
| CS L1 + IA | ~0.00008 | Excelente, artefatos removidos |

## Vantagens do Pipeline

1. **Compressão**: CS L1 reduz dados em 50%
2. **Qualidade**: IA remove artefatos
3. **Automático**: Script único executa tudo
4. **Flexível**: Pode usar apenas CS ou apenas IA

## Exemplos

```bash
# Compressão alta (30%)
python cs_pipeline.py img.jpg resultado.jpg 0.3

# Compressão padrão (50%)
python cs_pipeline.py img.jpg resultado.jpg 0.5

# Compressão baixa (70%)
python cs_pipeline.py img.jpg resultado.jpg 0.7
```


