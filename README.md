# Compressão de Imagens - Pesquisa

Projeto de pesquisa focado no estudo e implementação de técnicas de compressão de imagens utilizando métodos de álgebra linear e redes neurais.

## Sumário

- [Descrição](#descrição)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Métodos Implementados](#métodos-implementados)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Autor](#autor)

---

## Descrição

Este projeto explora diferentes abordagens para compressão de imagens digitais, combinando conceitos de álgebra linear clássica com técnicas modernas de aprendizado de máquina. O objetivo principal é investigar a eficiência e qualidade de reconstrução de cada método, analisando o trade-off entre taxa de compressão e fidelidade visual.

### Objetivos da Pesquisa

- Implementar manualmente a Decomposição em Valores Singulares (SVD) para compreensão profunda do método
- Desenvolver compressores híbridos que combinam SVD com autoencoders
- Explorar mapeamentos baseados em curvas de preenchimento espacial (Hilbert/Peano)
- Aplicar técnicas de visão computacional em imagens médicas (raio-X de pneumonia)

---

## Tecnologias Utilizadas

### Linguagens

- **Julia** (v1.x) - Implementações de álgebra linear e processamento de imagens
- **Python** (v3.8+) - Redes neurais e interfaces gráficas

### Bibliotecas Julia

| Biblioteca | Função |
|------------|--------|
| LinearAlgebra | Operações matriciais e decomposições |
| Images | Manipulação de imagens |
| ImageView | Visualização de imagens |
| CSV | Exportação de dados |
| DataFrames | Estruturação de dados |

### Bibliotecas Python

| Biblioteca | Função |
|------------|--------|
| PyTorch | Redes neurais (autoencoders) |
| NumPy | Computação numérica |
| OpenCV | Processamento de imagens |
| Pillow | Manipulação de imagens |
| Matplotlib | Visualização de dados |
| Streamlit | Interface web interativa |
| hilbertcurve | Curvas de Hilbert |

---

## Estrutura do Projeto

```
ImagesCompressing---Research/
│
├── CompressorSVD/                 # Módulo principal de compressão
│   ├── CompressorSVD.jl           # SVD manual para escala de cinza
│   ├── svd_compressor_rgb.jl      # SVD para imagens coloridas (RGB)
│   ├── autoencoder_compressor.py  # Compressor baseado em autoencoder
│   ├── hibrid_compressor.py       # Compressor híbrido (SVD + Autoencoder)
│   ├── peano_compressor.py        # Compressor usando curva de Hilbert
│   └── computer_vision.py         # Utilitários de visão computacional
│
├── PneumoProcImg/                 # Processamento de imagens médicas
│   └── main.py                    # Pré-processamento de raio-X
│
├── Teste/                         # Scripts de teste e experimentação
│   ├── array.jl
│   ├── teste.jl
│   └── ...
│
├── RedeNeural.py                  # Refinamento de imagens com autoencoder
├── svd.jl                         # Implementação base de SVD
├── images.jl                      # Utilitários de imagem
├── otimiza.jl                     # Algoritmos de otimização
├── requirements.txt               # Dependências Python
└── README.md                      # Este arquivo
```

---

## Métodos Implementados

### 1. Decomposição em Valores Singulares (SVD)

Implementação manual da SVD sem uso de funções prontas, seguindo o processo matemático:

1. Cálculo de A^T * A
2. Extração de autovalores e autovetores
3. Construção das matrizes U, Sigma e V
4. Reconstrução aproximada com k valores singulares

**Arquivos:** `CompressorSVD.jl`, `svd_compressor_rgb.jl`

### 2. Autoencoder Convolucional

Rede neural para refinamento de imagens pós-compressão:

- Encoder: Conv2D -> ReLU -> Conv2D -> ReLU
- Decoder: ConvTranspose2D -> ReLU -> ConvTranspose2D -> Sigmoid
- Função de perda: MSE (Mean Squared Error)

**Arquivo:** `RedeNeural.py`

### 3. Compressor com Curva de Hilbert

Mapeamento da imagem 2D para sequência 1D usando curva de preenchimento espacial, seguido de compressão com autoencoder linear.

- Preserva localidade espacial
- Interface interativa com Streamlit

**Arquivo:** `peano_compressor.py`

### 4. Processamento de Imagens Médicas

Pré-processamento de dataset de raio-X torácico para detecção de pneumonia:

- Extração automática de arquivos ZIP
- Redimensionamento padronizado (224x224)
- Organização por classes

**Arquivo:** `PneumoProcImg/main.py`

---

## Instalação

### Dependências Python

```bash
pip install -r requirements.txt
```

Para funcionalidades de redes neurais, instale também:

```bash
pip install torch torchvision streamlit hilbertcurve
```

### Dependências Julia

No REPL do Julia:

```julia
using Pkg
Pkg.add(["LinearAlgebra", "Images", "ImageView", "CSV", "DataFrames"])
```

---

## Como Usar

### Compressão SVD (Escala de Cinza)

```julia
include("CompressorSVD/CompressorSVD.jl")
```

Edite o arquivo para definir:
- Caminho da imagem de entrada
- Valor de k (número de valores singulares)
- Caminho da imagem de saída

### Compressão SVD (RGB)

```julia
include("CompressorSVD/svd_compressor_rgb.jl")
```

Ou utilize a função diretamente:

```julia
k = 50  # valores singulares
compressed_image, svd_r, svd_g, svd_b = compress_rgb_image("entrada.jpg", k, "saida.jpg")
```

### Compressor com Curva de Hilbert

```bash
streamlit run CompressorSVD/peano_compressor.py
```

Acesse a interface web no navegador e carregue uma imagem para compressão.

### Refinamento com Autoencoder

```bash
python RedeNeural.py
```

---

## Resultados

O projeto permite analisar:

- **Taxa de compressão**: Controlada pelo parâmetro k na SVD
- **Qualidade visual**: Comparação lado a lado entre original e comprimida
- **Métricas**: MSE (erro quadrático médio) durante treinamento do autoencoder

---

## Autores

Leonardo Arazo e Thadeu Senne


---

## Licença

Este projeto é destinado a fins acadêmicos e de pesquisa.
