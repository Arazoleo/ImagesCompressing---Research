export interface ImageFile {
  id: string;
  file: File;
  name: string;
  size: number;
  url: string;
  width?: number;
  height?: number;
  uploaded_at?: string;
}

export interface ProcessingResult {
  id: string;
  image_id: string;
  algorithm: AlgorithmType;
  status: 'pending' | 'processing' | 'completed' | 'error';
  result_url?: string;
  metrics?: {
    psnr?: number;
    ssim?: number;
    compression_ratio?: number;
    file_size?: number;
  };
  processing_time?: number;
  error_message?: string;
  created_at: string;
  job_id?: string;
}

export interface Metrics {
  psnr: number;
  ssim: number;
  mse: number;
  compressionRatio: number;
  fileSize: number;
  originalSize: number;
}

export type AlgorithmType =
  | 'svd'
  | 'compressed-sensing'
  | 'autoencoder'
  | 'hybrid'
  | 'hilbert'
  | 'fft'
  | 'wavelet'
  | 'pca'
  | 'ica';

export interface AlgorithmConfig {
  type: AlgorithmType;
  name: string;
  description: string;
  parameters: ParameterConfig[];
  category: 'decomposition' | 'compression' | 'neural' | 'hybrid';
  complexity: 'low' | 'medium' | 'high';
  color: string;
  icon: string;
}

export interface ParameterConfig {
  name: string;
  label: string;
  type: 'number' | 'slider' | 'select' | 'boolean';
  default: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: number | string; label: string }[];
  description?: string;
}

export interface ProcessingPipeline {
  id: string;
  name: string;
  steps: ProcessingStep[];
  createdAt: Date;
}

export interface ProcessingStep {
  algorithm: AlgorithmType;
  parameters: Record<string, number | string | boolean>;
  order: number;
}

export interface ComparisonResult {
  id: string;
  original: ImageFile;
  results: ProcessingResult[];
  bestAlgorithm: AlgorithmType;
  createdAt: Date;
}

export type AlgorithmParams = Record<string, number | string | boolean>;

export interface ProcessingJobStatus {
  job_id: string;
  total_images: number;
  total_algorithms: number;
  estimated_time: number;
  progress: number;
  status: string;
  results: Array<{
    id: string;
    image_id: string;
    algorithm: AlgorithmType;
    status: 'pending' | 'processing' | 'completed' | 'error';
    result_url?: string;
    metrics?: {
      psnr?: number;
      ssim?: number;
      compression_ratio?: number;
    };
    processing_time?: number;
    error_message?: string;
    created_at: string;
  }>;
}
