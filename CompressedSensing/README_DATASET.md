# Treinamento do Refinador CS em Dataset

Este guia explica como treinar o modelo de refinamento para funcionar em **qualquer imagem**.

## Opção Rápida: Usar Dataset Pronto

```bash
# Baixa Set5 (5 imagens) - rápido para testes
python download_dataset_simple.py

# Ou Set14 (14 imagens) - mais completo
python download_dataset.py
# Escolha opção 1
```

Depois:
```bash
# Gerar imagens CS
python generate_cs_images.py

# Treinar
python cs_train_dataset.py
```

## Estrutura Necessária

```
CompressedSensing/
├── dataset/
│   ├── original/              # Imagens originais
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── cs_reconstructed/      # Imagens comprimidas pelo CS L1
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
├── cs_train_dataset.py        # Script de treinamento
├── cs_apply_model.py          # Script para aplicar em nova imagem
└── cs_refiner_model.pth       # Modelo treinado (gerado)
```

## Passo 1: Preparar Dataset

### Opção A: Usar imagens existentes

```bash
# Cria estrutura e copia imagens
python prepare_dataset.py ./minhas_imagens
```

### Opção B: Criar manualmente

1. Crie as pastas:
```bash
mkdir -p dataset/original
mkdir -p dataset/cs_reconstructed
```

2. Coloque imagens originais em `dataset/original/`

3. Aplique CS L1 em cada imagem e salve em `dataset/cs_reconstructed/`

## Passo 2: Gerar Imagens CS (se necessário)

Se você já tem imagens comprimidas, pule este passo.

Para gerar imagens CS L1 de todas as imagens originais, você pode criar um script Julia ou Python que processa todas as imagens da pasta.

## Passo 3: Treinar Modelo

```bash
python cs_train_dataset.py
```

O modelo será salvo em `cs_refiner_model.pth`

## Passo 4: Aplicar em Nova Imagem

```bash
python cs_apply_model.py cs_refiner_model.pth img_reconstructed_l1.jpg img_refined.jpg
```

## Parâmetros de Treinamento

Edite `cs_train_dataset.py` para ajustar:
- `num_epochs`: Número de épocas (padrão: 50)
- `batch_size`: Tamanho do batch (padrão: 4)
- `lr`: Learning rate (padrão: 0.0001)

## Recomendações

- **Mínimo**: 10-20 imagens para começar
- **Ideal**: 50-100+ imagens para melhor generalização
- **Tamanho**: Imagens serão redimensionadas para 256x256 durante treinamento
- **Variedade**: Use imagens variadas (diferentes tipos, iluminação, etc.)

