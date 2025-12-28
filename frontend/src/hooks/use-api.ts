/**
 * Custom hooks para integração com a API
 */

import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";
import { AlgorithmType, ImageFile } from "@/lib/types";

// Hook para buscar algoritmos disponíveis
export function useAlgorithms() {
  return useQuery({
    queryKey: ["algorithms"],
    queryFn: () => apiClient.getAvailableAlgorithms(),
    staleTime: 5 * 60 * 1000, // 5 minutos
  });
}

// Hook para buscar categorias de algoritmos
export function useAlgorithmCategories() {
  return useQuery({
    queryKey: ["algorithm-categories"],
    queryFn: () => apiClient.getAlgorithmCategories(),
    staleTime: 5 * 60 * 1000, // 5 minutos
  });
}

// Hook para buscar imagens carregadas
export function useImages(limit = 50, offset = 0) {
  return useQuery({
    queryKey: ["images", limit, offset],
    queryFn: () => apiClient.getImages(limit, offset),
  });
}

// Hook para upload de imagem
export function useUploadImage() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => apiClient.uploadImage(file),
    onSuccess: () => {
      // Invalidar cache das imagens
      queryClient.invalidateQueries({ queryKey: ["images"] });
    },
  });
}

// Hook para processamento de imagens
export function useProcessImages() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: {
      imageIds: string[];
      algorithms: { type: AlgorithmType; parameters: Record<string, any> }[];
    }) => apiClient.processImages(data.imageIds, data.algorithms),
  });
}

// Hook para buscar status do processamento
export function useProcessingStatus(jobId: string, enabled = true) {
  return useQuery({
    queryKey: ["processing", jobId],
    queryFn: () => apiClient.getProcessingStatus(jobId),
    enabled: !!jobId && enabled,
    refetchInterval: 2000,
  });
}

// Hook para buscar histórico de processamento
export function useProcessingHistory(limit = 50, offset = 0) {
  return useQuery({
    queryKey: ["processing-history", limit, offset],
    queryFn: () => apiClient.getProcessingHistory(limit, offset),
  });
}

// Hook para buscar resultados de um job específico
export function useProcessingResults(jobId: string | null, enabled = true) {
  return useQuery({
    queryKey: ["processing-results", jobId],
    queryFn: () => apiClient.getProcessingResults(jobId!),
    enabled: !!jobId && enabled,
  });
}

// Hook para buscar estatísticas
export function useProcessingStats() {
  return useQuery({
    queryKey: ["processing-stats"],
    queryFn: () => apiClient.getProcessingStats(),
    staleTime: 30 * 1000, // 30 segundos
  });
}

// Hook para WebSocket de processamento em tempo real
export function useProcessingWebSocket(jobId: string | null, enabled = true) {
  const [progress, setProgress] = useState<any[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!jobId || !enabled) return;

    const ws = apiClient.connectToProcessingUpdates(jobId, (data) => {
      setProgress(prev => {
        // Atualizar progresso existente ou adicionar novo
        const existingIndex = prev.findIndex(p => p.algorithm === data.algorithm);
        if (existingIndex >= 0) {
          const updated = [...prev];
          updated[existingIndex] = { ...updated[existingIndex], ...data };
          return updated;
        } else {
          return [...prev, data];
        }
      });
    });

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [jobId, enabled]);

  return { progress, isConnected };
}
