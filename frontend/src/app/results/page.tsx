"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ImageViewer } from "@/components/image/image-viewer";
import { BarChart3, Download, Share, TrendingUp, Trophy, Zap, Clock, FileImage, Loader2, AlertCircle } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useProcessingHistory, useProcessingResults } from "@/hooks/use-api";
import { useImages } from "@/hooks/use-api";

export default function ResultsPage() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(0);
  
  const { data: history, isLoading: historyLoading } = useProcessingHistory(20);
  const { data: results, isLoading: resultsLoading } = useProcessingResults(selectedJobId, !!selectedJobId);
  const { data: images } = useImages();

  // Selecionar o job mais recente automaticamente
  const latestJob = useMemo(() => {
    if (!history || history.length === 0) return null;
    // Agrupar por job_id e pegar o mais recente
    const jobsMap = new Map<string, any>();
    history.forEach((item: any) => {
      const jobId = item.job_id || item.id?.split('-')[0];
      if (jobId && !jobsMap.has(jobId)) {
        jobsMap.set(jobId, item);
      }
    });
    return Array.from(jobsMap.values())[0];
  }, [history]);

  // Atualizar jobId selecionado quando houver um job mais recente
  useMemo(() => {
    if (latestJob && !selectedJobId) {
      const jobId = latestJob.job_id || latestJob.id?.split('-')[0];
      if (jobId) setSelectedJobId(jobId);
    }
  }, [latestJob, selectedJobId]);

  // Processar resultados para exibição
  const processedImages = useMemo(() => {
    if (!results || results.length === 0) return [];
    
    return results
      .filter((r: any) => r.status === 'completed' && r.result_url)
      .map((r: any) => ({
        url: r.result_url,
        name: `${r.algorithm.toUpperCase()} Compression`,
        algorithm: r.algorithm,
        metrics: {
          psnr: r.metrics?.psnr || 0,
          ssim: r.metrics?.ssim || 0,
          compressionRatio: r.metrics?.compression_ratio || 0,
          fileSize: r.metrics?.file_size || 0,
        },
        processingTime: r.processing_time || 0,
      }));
  }, [results]);

  // Buscar imagem original
  const originalImage = useMemo(() => {
    if (!results || results.length === 0 || !images) {
      return { url: '/api/placeholder/512/512', name: 'N/A', width: 512, height: 512 };
    }
    
    const firstResult = results[0];
    const image = images.find((img: any) => img.id === firstResult.image_id);
    
    if (image) {
      return {
        url: image.url,
        name: image.name || 'Imagem',
        width: image.width || 512,
        height: image.height || 512,
      };
    }
    
    return { url: '/api/placeholder/512/512', name: 'N/A', width: 512, height: 512 };
  }, [results, images]);

  const bestAlgorithm = useMemo(() => {
    if (processedImages.length === 0) return null;
    return processedImages.reduce((best, current, index) => 
      (current.metrics.psnr > best.metrics.psnr) ? { ...current, index } : best, 
      { ...processedImages[0], index: 0 }
    );
  }, [processedImages]);

  // Agrupar jobs únicos do histórico
  const uniqueJobs = useMemo(() => {
    if (!history) return [];
    const jobsMap = new Map<string, any>();
    history.forEach((item: any) => {
      const jobId = item.job_id || item.id?.split('-')[0];
      if (jobId && !jobsMap.has(jobId)) {
        jobsMap.set(jobId, {
          jobId,
          createdAt: item.created_at,
          algorithm: item.algorithm,
        });
      }
    });
    return Array.from(jobsMap.values()).sort((a, b) => 
      new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );
  }, [history]);

  if (historyLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="ml-2 text-lg text-muted-foreground">Carregando resultados...</p>
      </div>
    );
  }

  if (!history || history.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
        <FileImage className="h-12 w-12 mb-4 opacity-50" />
        <p className="text-xl font-medium">Nenhum resultado encontrado</p>
        <p className="text-sm">Processe algumas imagens para ver os resultados aqui.</p>
      </div>
    );
  }

  return (
    <motion.div className="flex-1 space-y-8 p-6 md:p-8" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Resultados do Processamento</h1>
          <p className="text-muted-foreground mt-1">Análise detalhada dos algoritmos aplicados</p>
        </div>
        <div className="flex items-center gap-3">
          <Select value={selectedJobId || ""} onValueChange={setSelectedJobId}>
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Selecione um job" />
            </SelectTrigger>
            <SelectContent>
              {uniqueJobs.map((job: any) => (
                <SelectItem key={job.jobId} value={job.jobId}>
                  {job.algorithm?.toUpperCase()} - {new Date(job.createdAt).toLocaleString()}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" className="rounded-xl"><Share className="h-4 w-4 mr-2" />Compartilhar</Button>
          <Button className="rounded-xl bg-gradient-to-r from-primary to-primary/80"><Download className="h-4 w-4 mr-2" />Exportar Relatório</Button>
        </div>
      </div>

      {resultsLoading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="ml-2 text-lg text-muted-foreground">Carregando resultados do job...</p>
        </div>
      ) : processedImages.length === 0 ? (
        <Card>
          <CardContent className="p-12 text-center">
            <AlertCircle className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
            <p className="text-lg font-medium mb-2">Nenhum resultado disponível</p>
            <p className="text-sm text-muted-foreground">Selecione um job com resultados processados.</p>
          </CardContent>
        </Card>
      ) : (
        <>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Image Viewer */}
        <div className="lg:col-span-2">
          <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
            <CardContent className="p-6">
              <ImageViewer originalImage={originalImage} processedImages={processedImages} />
            </CardContent>
          </Card>
        </div>

        {/* Results Analysis */}
        <div className="space-y-6">
          {/* Best Algorithm */}
          <Card className="relative overflow-hidden border-border/50 bg-gradient-to-br from-emerald-500/5 via-green-500/5 to-transparent">
            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-emerald-500/20 to-transparent rounded-full -translate-y-16 translate-x-16 blur-2xl" />
            <CardHeader className="relative">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-green-500 flex items-center justify-center shadow-lg">
                  <Trophy className="h-5 w-5 text-white" />
                </div>
                <CardTitle className="text-emerald-700 dark:text-emerald-400">Melhor Algoritmo</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="relative">
              {bestAlgorithm ? (
                <>
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-semibold text-lg">{bestAlgorithm.name}</span>
                    <Badge className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-0">Recomendado</Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-muted/30 rounded-xl p-3 text-center">
                      <div className="text-2xl font-bold text-emerald-600">{bestAlgorithm.metrics.psnr.toFixed(2)}</div>
                      <div className="text-xs text-muted-foreground">PSNR (dB)</div>
                    </div>
                    <div className="bg-muted/30 rounded-xl p-3 text-center">
                      <div className="text-2xl font-bold text-emerald-600">{bestAlgorithm.metrics.ssim.toFixed(3)}</div>
                      <div className="text-xs text-muted-foreground">SSIM</div>
                    </div>
                  </div>
                </>
              ) : (
                <p className="text-muted-foreground text-center py-4">Nenhum algoritmo disponível</p>
              )}
            </CardContent>
          </Card>

          {/* Algorithm Comparison */}
          <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center shadow-lg">
                  <BarChart3 className="h-5 w-5 text-white" />
                </div>
                <CardTitle>Comparação</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {processedImages.map((algorithm, index) => (
                  <motion.div
                    key={index}
                    whileHover={{ scale: 1.02 }}
                    className={cn(
                      "p-4 rounded-xl border-2 cursor-pointer transition-all",
                      selectedAlgorithm === index
                        ? "border-primary bg-primary/5"
                        : "border-border/50 hover:border-primary/50"
                    )}
                    onClick={() => setSelectedAlgorithm(index)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{algorithm.name}</span>
                      <Badge variant={selectedAlgorithm === index ? "default" : "secondary"} className="rounded-lg">
                        {algorithm.metrics.psnr} dB
                      </Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs text-muted-foreground">
                      <span>PSNR: {algorithm.metrics.psnr.toFixed(2)}</span>
                      <span>SSIM: {algorithm.metrics.ssim.toFixed(3)}</span>
                      <span>Ratio: {algorithm.metrics.compressionRatio.toFixed(2)}</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Processing Details */}
          <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center shadow-lg">
                  <Zap className="h-5 w-5 text-white" />
                </div>
                <CardTitle>Detalhes</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { label: "Imagem Original", value: `${originalImage.width}×${originalImage.height}`, icon: FileImage },
                { label: "Formatos Processados", value: processedImages.length.toString(), icon: BarChart3 },
                { label: "Tempo Total", value: processedImages.length > 0 ? `${processedImages.reduce((sum, alg) => sum + (alg.processingTime || 0), 0).toFixed(2)}s` : "N/A", icon: Clock },
                { label: "Melhor PSNR", value: processedImages.length > 0 ? `${Math.max(...processedImages.map(a => a.metrics.psnr)).toFixed(2)} dB` : "N/A", icon: TrendingUp },
              ].map((item, i) => (
                <div key={i} className="flex items-center justify-between text-sm bg-muted/20 rounded-lg px-3 py-2">
                  <span className="flex items-center gap-2 text-muted-foreground">
                    <item.icon className="h-4 w-4" />{item.label}
                  </span>
                  <span className="font-medium">{item.value}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Detailed Metrics */}
      <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shadow-lg">
              <BarChart3 className="h-5 w-5 text-white" />
            </div>
            <div>
              <CardTitle>Análise Detalhada de Métricas</CardTitle>
              <CardDescription>Comparação completa de todas as métricas</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="quality">
            <TabsList className="bg-muted/30 rounded-xl p-1">
              <TabsTrigger value="quality" className="rounded-lg">Qualidade</TabsTrigger>
              <TabsTrigger value="compression" className="rounded-lg">Compressão</TabsTrigger>
              <TabsTrigger value="performance" className="rounded-lg">Performance</TabsTrigger>
            </TabsList>
            <TabsContent value="quality" className="mt-6">
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-4">
                  <h4 className="font-semibold flex items-center gap-2"><TrendingUp className="h-4 w-4 text-blue-500" />PSNR (Peak Signal-to-Noise Ratio)</h4>
                  {processedImages.map((alg, i) => (
                    <div key={i} className="flex justify-between text-sm bg-muted/20 rounded-lg px-3 py-2">
                      <span>{alg.name}</span><span className="font-medium">{alg.metrics.psnr.toFixed(2)} dB</span>
                    </div>
                  ))}
                </div>
                <div className="space-y-4">
                  <h4 className="font-semibold flex items-center gap-2"><TrendingUp className="h-4 w-4 text-violet-500" />SSIM (Structural Similarity)</h4>
                  {processedImages.map((alg, i) => (
                    <div key={i} className="flex justify-between text-sm bg-muted/20 rounded-lg px-3 py-2">
                      <span>{alg.name}</span><span className="font-medium">{alg.metrics.ssim.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>
            <TabsContent value="compression" className="mt-6">
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-4">
                  <h4 className="font-semibold">Taxa de Compressão</h4>
                  {processedImages.map((alg, i) => (
                    <div key={i} className="flex justify-between text-sm bg-muted/20 rounded-lg px-3 py-2">
                      <span>{alg.name}</span><span className="font-medium">{(alg.metrics.compressionRatio * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
                <div className="space-y-4">
                  <h4 className="font-semibold">Tamanho do Arquivo</h4>
                  {processedImages.map((alg, i) => (
                    <div key={i} className="flex justify-between text-sm bg-muted/20 rounded-lg px-3 py-2">
                      <span>{alg.name}</span><span className="font-medium">{alg.metrics.fileSize} KB</span>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>
            <TabsContent value="performance" className="mt-6">
              <div className="text-center py-12">
                <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="text-muted-foreground">Métricas de performance em desenvolvimento</p>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
        </>
      )}
    </motion.div>
  );
}
