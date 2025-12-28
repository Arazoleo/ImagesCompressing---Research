import { AlgorithmConfig, AlgorithmType } from './types';

export const ALGORITHMS: Record<string, AlgorithmConfig> = {
  svd: {
    type: 'svd',
    name: 'SVD Compression',
    description: 'Decomposição em Valores Singulares para compressão matricial',
    category: 'decomposition',
    complexity: 'low',
    color: 'bg-blue-500',
    icon: '🧮',
    parameters: [
      {
        name: 'k',
        label: 'Valores Singulares',
        type: 'slider',
        default: 50,
        min: 1,
        max: 200,
        step: 1,
        description: 'Número de valores singulares para manter'
      }
    ]
  },
  'compressed-sensing': {
    type: 'compressed-sensing',
    name: 'Compressed Sensing',
    description: 'Compressão baseada em amostragem esparsa com reconstrução L1',
    category: 'compression',
    complexity: 'high',
    color: 'bg-green-500',
    icon: '📡',
    parameters: [
      {
        name: 'samplingRate',
        label: 'Taxa de Amostragem',
        type: 'slider',
        default: 0.5,
        min: 0.1,
        max: 0.9,
        step: 0.1,
        description: 'Porcentagem de amostras coletadas'
      },
      {
        name: 'sparsifyingBasis',
        label: 'Base Esparsificante',
        type: 'select',
        default: 'dct',
        options: [
          { value: 'dct', label: 'DCT (Cosseno)' },
          { value: 'wavelet', label: 'Wavelet' },
          { value: 'fft', label: 'FFT' }
        ],
        description: 'Transformada para representação esparsa'
      }
    ]
  },
  autoencoder: {
    type: 'autoencoder',
    name: 'Autoencoder',
    description: 'Rede neural autoencoder para compressão aprendida',
    category: 'neural',
    complexity: 'medium',
    color: 'bg-purple-500',
    icon: '🧠',
    parameters: [
      {
        name: 'latentDim',
        label: 'Dimensão Latente',
        type: 'slider',
        default: 64,
        min: 16,
        max: 256,
        step: 16,
        description: 'Dimensão do espaço latente'
      },
      {
        name: 'epochs',
        label: 'Épocas',
        type: 'number',
        default: 100,
        min: 10,
        max: 1000,
        description: 'Número de épocas de treinamento'
      }
    ]
  },
  hybrid: {
    type: 'hybrid',
    name: 'SVD + Autoencoder',
    description: 'Compressão híbrida combinando SVD com refinamento neural',
    category: 'hybrid',
    complexity: 'high',
    color: 'bg-orange-500',
    icon: '🔄',
    parameters: [
      {
        name: 'svdK',
        label: 'Valores Singulares SVD',
        type: 'slider',
        default: 30,
        min: 5,
        max: 100,
        step: 5,
        description: 'Valores singulares para pré-processamento'
      },
      {
        name: 'refinementEpochs',
        label: 'Épocas Refinamento',
        type: 'number',
        default: 50,
        min: 10,
        max: 200,
        description: 'Épocas para refinamento neural'
      }
    ]
  },
  hilbert: {
    type: 'hilbert',
    name: 'Curva de Hilbert',
    description: 'Mapeamento 2D→1D usando curva de Hilbert com autoencoder',
    category: 'hybrid',
    complexity: 'medium',
    color: 'bg-cyan-500',
    icon: '🌀',
    parameters: [
      {
        name: 'curveOrder',
        label: 'Ordem da Curva',
        type: 'slider',
        default: 7,
        min: 4,
        max: 10,
        step: 1,
        description: 'Ordem da curva de Hilbert (2^ordem = tamanho)'
      },
      {
        name: 'bottleneckSize',
        label: 'Tamanho do Gargalo',
        type: 'slider',
        default: 128,
        min: 32,
        max: 512,
        step: 32,
        description: 'Dimensão do gargalo na sequência 1D'
      }
    ]
  },
  fft: {
    type: 'fft',
    name: 'FFT Compression',
    description: 'Compressão baseada em Transformada Rápida de Fourier',
    category: 'decomposition',
    complexity: 'low',
    color: 'bg-indigo-500',
    icon: '🌊',
    parameters: [
      {
        name: 'frequencyCutoff',
        label: 'Corte de Frequência',
        type: 'slider',
        default: 0.3,
        min: 0.1,
        max: 0.8,
        step: 0.05,
        description: 'Porcentagem de frequências mantidas'
      }
    ]
  },
  wavelet: {
    type: 'wavelet',
    name: 'Wavelet Transform',
    description: 'Compressão multiresolução usando wavelets',
    category: 'decomposition',
    complexity: 'medium',
    color: 'bg-teal-500',
    icon: '🌊',
    parameters: [
      {
        name: 'waveletType',
        label: 'Tipo de Wavelet',
        type: 'select',
        default: 'haar',
        options: [
          { value: 'haar', label: 'Haar' },
          { value: 'db4', label: 'Daubechies 4' },
          { value: 'sym4', label: 'Symlet 4' }
        ],
        description: 'Família de wavelet utilizada'
      },
      {
        name: 'threshold',
        label: 'Threshold',
        type: 'slider',
        default: 0.1,
        min: 0.01,
        max: 0.5,
        step: 0.01,
        description: 'Threshold para thresholding soft'
      }
    ]
  },
  pca: {
    type: 'pca',
    name: 'PCA Compression',
    description: 'Compressão via Análise de Componentes Principais',
    category: 'decomposition',
    complexity: 'low',
    color: 'bg-pink-500',
    icon: '📊',
    parameters: [
      {
        name: 'components',
        label: 'Componentes',
        type: 'slider',
        default: 20,
        min: 5,
        max: 100,
        step: 5,
        description: 'Número de componentes principais'
      }
    ]
  },
  ica: {
    type: 'ica',
    name: 'ICA Compression',
    description: 'Compressão via Análise de Componentes Independentes',
    category: 'decomposition',
    complexity: 'high',
    color: 'bg-red-500',
    icon: '🎯',
    parameters: [
      {
        name: 'components',
        label: 'Componentes',
        type: 'slider',
        default: 15,
        min: 5,
        max: 50,
        step: 5,
        description: 'Número de componentes independentes'
      },
      {
        name: 'maxIter',
        label: 'Iterações Máximas',
        type: 'number',
        default: 1000,
        min: 100,
        max: 5000,
        description: 'Máximo de iterações para convergência'
      }
    ]
  }
};

export type { AlgorithmConfig, AlgorithmType };

export const ALGORITHM_CATEGORIES = {
  decomposition: {
    name: 'Decomposições Matriciais',
    description: 'Métodos baseados em decomposição de matrizes',
    color: 'bg-blue-100 text-blue-800'
  },
  compression: {
    name: 'Compressão Clássica',
    description: 'Técnicas tradicionais de compressão',
    color: 'bg-green-100 text-green-800'
  },
  neural: {
    name: 'Redes Neurais',
    description: 'Métodos baseados em aprendizado profundo',
    color: 'bg-purple-100 text-purple-800'
  },
  hybrid: {
    name: 'Métodos Híbridos',
    description: 'Combinação de múltiplas abordagens',
    color: 'bg-orange-100 text-orange-800'
  }
};
