import { ImageFile, ProcessingResult, ComparisonResult, AlgorithmType, AlgorithmParams, ProcessingJobStatus } from './types';
import { config } from './config';

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl = config.API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async uploadImage(file: File): Promise<ImageFile> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/images/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return response.json();
  }

  async processImage(
    imageId: string,
    algorithm: AlgorithmType,
    parameters: AlgorithmParams
  ): Promise<ProcessingResult> {
    const response = await fetch(`${this.baseUrl}/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        imageId,
        algorithm,
        parameters,
      }),
    });

    if (!response.ok) {
      throw new Error(`Processing failed: ${response.statusText}`);
    }

    return response.json();
  }

  async processImages(
    imageIds: string[],
    algorithms: { type: AlgorithmType; parameters: Record<string, any> }[]
  ): Promise<{ job_id: string; total_images: number; total_algorithms: number }> {
    const response = await fetch(`${this.baseUrl}/processing`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_ids: imageIds,
        algorithms: algorithms.map(alg => ({
          type: alg.type,
          parameters: alg.parameters
        })),
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to start processing: ${response.statusText}`);
    }

    return response.json();
  }

  async compareAlgorithms(
    imageId: string,
    algorithms: { type: AlgorithmType; parameters: AlgorithmParams }[]
  ): Promise<ComparisonResult> {
    const response = await fetch(`${this.baseUrl}/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        imageId,
        algorithms,
      }),
    });

    if (!response.ok) {
      throw new Error(`Comparison failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getMetrics(imageId: string, resultId: string): Promise<Record<string, number>> {
    const response = await fetch(`${this.baseUrl}/metrics/${imageId}/${resultId}`);

    if (!response.ok) {
      throw new Error(`Failed to fetch metrics: ${response.statusText}`);
    }

    return response.json();
  }

  // --- Images API ---

  async getImages(limit = 50, offset = 0): Promise<ImageFile[]> {
    const response = await fetch(`${this.baseUrl}/images?limit=${limit}&offset=${offset}`);

    if (!response.ok) {
      throw new Error(`Failed to fetch images: ${response.statusText}`);
    }

    return response.json();
  }

  async deleteImage(imageId: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${this.baseUrl}/images/${imageId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error(`Failed to delete image: ${response.statusText}`);
    }

    return response.json();
  }

  // --- Algorithms API ---

  async getAvailableAlgorithms(): Promise<Record<string, any>> {
    const response = await fetch(`${this.baseUrl}/algorithms`);

    if (!response.ok) {
      throw new Error(`Failed to fetch algorithms: ${response.statusText}`);
    }

    return response.json();
  }

  async getAlgorithmCategories(): Promise<Record<string, any>> {
    const response = await fetch(`${this.baseUrl}/algorithms/categories`);

    if (!response.ok) {
      throw new Error(`Failed to fetch algorithm categories: ${response.statusText}`);
    }

    return response.json();
  }

  async validateAlgorithmParameters(
    algorithmType: AlgorithmType,
    parameters: Record<string, any>
  ): Promise<{ valid: boolean; errors: string[] }> {
    const response = await fetch(`${this.baseUrl}/algorithms/${algorithmType}/validate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(parameters),
    });

    if (!response.ok) {
      throw new Error(`Failed to validate parameters: ${response.statusText}`);
    }

    return response.json();
  }

  // --- Processing History ---

  async getProcessingStatus(jobId: string): Promise<ProcessingJobStatus> {
    const response = await fetch(`${this.baseUrl}/processing/${jobId}/status`);

    if (!response.ok) {
      throw new Error(`Failed to fetch processing status: ${response.statusText}`);
    }

    return response.json();
  }

  async getProcessingResults(jobId: string): Promise<ProcessingResult[]> {
    const response = await fetch(`${this.baseUrl}/processing/${jobId}/results`);

    if (!response.ok) {
      throw new Error(`Failed to fetch processing results: ${response.statusText}`);
    }

    return response.json();
  }

  async getProcessingHistory(limit = 50, offset = 0): Promise<ProcessingResult[]> {
    const response = await fetch(`${this.baseUrl}/processing/history?limit=${limit}&offset=${offset}`);

    if (!response.ok) {
      throw new Error(`Failed to fetch processing history: ${response.statusText}`);
    }

    return response.json();
  }

  async getProcessingStats(): Promise<Record<string, any>> {
    const response = await fetch(`${this.baseUrl}/processing/stats`);

    if (!response.ok) {
      throw new Error(`Failed to fetch processing stats: ${response.statusText}`);
    }

    return response.json();
  }

  // --- WebSocket connection for real-time processing updates ---
  connectToProcessingUpdates(jobId: string, onMessage: (data: any) => void): WebSocket {
    const wsUrl = config.WS_BASE_URL.replace('http', 'ws');
    const ws = new WebSocket(`${wsUrl}/ws/processing/${jobId}`);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return ws;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Utility functions for image handling
export const createImageUrl = (file: File): Promise<string> => {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    resolve(url);
  });
};

export const getImageDimensions = (file: File): Promise<{ width: number; height: number }> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      resolve({ width: img.width, height: img.height });
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
};

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const formatProcessingTime = (ms: number): string => {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};
