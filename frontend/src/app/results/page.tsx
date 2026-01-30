"use client";

import { useState, useMemo, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ImageViewer } from "@/components/image/image-viewer";
import { BarChart3, Download, Share, TrendingUp, Award, Zap, Clock, FileImage, Loader2, AlertCircle, Box } from "lucide-react";
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
    const jobsMap = new Map<string, any>();
    history.forEach((item: any) => {
      const jobId = item.job_id;
      if (jobId && typeof jobId === 'string' && jobId.length > 0 && !jobsMap.has(jobId)) {
        jobsMap.set(jobId, item);
      }
    });
    const jobs = Array.from(jobsMap.values());
    if (jobs.length === 0) return null;
    jobs.sort((a, b) => {
      const dateA = new Date(a.created_at || 0).getTime();
      const dateB = new Date(b.created_at || 0).getTime();
      return dateB - dateA;
    });
    return jobs[0];
  }, [history]);

  useEffect(() => {
    if (latestJob && !selectedJobId) {
      const jobId = latestJob.job_id;
      if (jobId && typeof jobId === 'string' && jobId.length > 0) {
        setSelectedJobId(jobId);
      }
    }
  }, [latestJob, selectedJobId]);

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

  const uniqueJobs = useMemo(() => {
    if (!history) return [];
    const jobsMap = new Map<string, any>();
    history.forEach((item: any) => {
      const jobId = item.job_id;
      if (jobId && typeof jobId === 'string' && jobId.length > 0 && !jobsMap.has(jobId)) {
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
      <div className="flex flex-col items-center justify-center h-[60vh]">
        <Loader2 className="h-8 w-8 animate-spin text-foreground mb-4" strokeWidth={1.5} />
        <p className="text-muted-foreground">Carregando resultados...</p>
      </div>
    );
  }

  if (!history || history.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh]">
        <div className="p-4 rounded-2xl bg-accent border border-border mb-4">
          <FileImage className="h-8 w-8 text-muted-foreground" strokeWidth={1.5} />
        </div>
        <p className="text-xl font-semibold mb-2">Nenhum resultado</p>
        <p className="text-sm text-muted-foreground">Processe algumas imagens para ver os resultados aqui.</p>
      </div>
    );
  }

  return (
    <motion.div 
      className="flex-1 space-y-8 p-6 md:p-10" 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }}
    >
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-1">
          <p className="text-caption">Análise</p>
          <h1 className="text-headline">Resultados</h1>
          <p className="text-muted-foreground">Análise detalhada dos algoritmos aplicados</p>
        </div>
        <div className="flex items-center gap-3">
          <Select 
            value={selectedJobId && typeof selectedJobId === 'string' && selectedJobId.length > 0 ? selectedJobId : ""} 
            onValueChange={(value) => {
              if (value && typeof value === 'string' && value.length > 0) {
                setSelectedJobId(value);
              }
            }}
          >
            <SelectTrigger className="w-[200px] bg-accent/50 border-border rounded-xl">
              <SelectValue placeholder="Selecione um job" />
            </SelectTrigger>
            <SelectContent className="rounded-xl">
              {uniqueJobs.map((job: any) => (
                <SelectItem key={job.jobId} value={job.jobId} className="rounded-lg">
                  {job.algorithm?.toUpperCase()} - {new Date(job.createdAt).toLocaleString()}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" className="rounded-xl border-border hover:bg-accent">
            <Share className="h-4 w-4 mr-2" strokeWidth={1.5} />
            Compartilhar
          </Button>
          <Button className="rounded-xl bg-foreground text-background hover:bg-foreground/90">
            <Download className="h-4 w-4 mr-2" strokeWidth={1.5} />
            Exportar
          </Button>
        </div>
      </div>

      {resultsLoading ? (
        <div className="flex flex-col items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-foreground mb-4" strokeWidth={1.5} />
          <p className="text-muted-foreground">Carregando resultados do job...</p>
        </div>
      ) : !selectedJobId ? (
        <Card className="border-border bg-card/50">
          <CardContent className="p-16 text-center">
            <div className="p-4 rounded-2xl bg-accent border border-border inline-block mb-4">
              <AlertCircle className="h-8 w-8 text-muted-foreground" strokeWidth={1.5} />
            </div>
            <p className="text-lg font-semibold mb-2">Nenhum job selecionado</p>
            <p className="text-sm text-muted-foreground max-w-sm mx-auto">
              Selecione um job no menu acima ou processe uma nova imagem.
            </p>
          </CardContent>
        </Card>
      ) : processedImages.length === 0 ? (
        <Card className="border-border bg-card/50">
          <CardContent className="p-16 text-center">
            <div className="p-4 rounded-2xl bg-accent border border-border inline-block mb-4">
              <AlertCircle className="h-8 w-8 text-muted-foreground" strokeWidth={1.5} />
            </div>
            <p className="text-lg font-semibold mb-2">Nenhum resultado disponível</p>
            <p className="text-sm text-muted-foreground max-w-sm mx-auto">
              Este job não possui resultados processados. O job pode ter sido perdido após reiniciar o servidor.
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid gap-6 lg:grid-cols-3">
            {/* Image Viewer */}
            <div className="lg:col-span-2">
              <Card className="border-border bg-card/50">
                <CardContent className="p-6">
                  <ImageViewer originalImage={originalImage} processedImages={processedImages} />
                </CardContent>
              </Card>
            </div>

            {/* Results Analysis */}
            <div className="space-y-6">
              {/* Best Algorithm */}
              <Card className="border-border bg-card/50 overflow-hidden">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-foreground">
                      <Award className="h-5 w-5 text-background" strokeWidth={1.5} />
                    </div>
                    <CardTitle>Melhor Algoritmo</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  {bestAlgorithm ? (
                    <>
                      <div className="flex items-center justify-between mb-4">
                        <span className="font-semibold text-lg">{bestAlgorithm.name}</span>
                        <Badge className="bg-foreground text-background border-0 rounded-md">
                          Recomendado
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-accent rounded-xl p-4 text-center border border-border">
                          <div className="text-2xl font-semibold">{bestAlgorithm.metrics.psnr.toFixed(2)}</div>
                          <div className="text-xs text-muted-foreground mt-1">PSNR (dB)</div>
                        </div>
                        <div className="bg-accent rounded-xl p-4 text-center border border-border">
                          <div className="text-2xl font-semibold">{bestAlgorithm.metrics.ssim.toFixed(3)}</div>
                          <div className="text-xs text-muted-foreground mt-1">SSIM</div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <p className="text-muted-foreground text-center py-4">Nenhum algoritmo disponível</p>
                  )}
                </CardContent>
              </Card>

              {/* Algorithm Comparison */}
              <Card className="border-border bg-card/50">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-accent border border-border">
                      <BarChart3 className="h-5 w-5" strokeWidth={1.5} />
                    </div>
                    <CardTitle>Comparação</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {processedImages.map((algorithm, index) => (
                      <motion.div
                        key={index}
                        whileHover={{ scale: 1.01 }}
                        className={cn(
                          "p-4 rounded-xl border cursor-pointer transition-all duration-300",
                          selectedAlgorithm === index
                            ? "border-foreground bg-accent"
                            : "border-border hover:border-foreground/30"
                        )}
                        onClick={() => setSelectedAlgorithm(index)}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">{algorithm.name}</span>
                          <Badge 
                            variant={selectedAlgorithm === index ? "default" : "secondary"} 
                            className={cn(
                              "rounded-md",
                              selectedAlgorithm === index 
                                ? "bg-foreground text-background" 
                                : "bg-accent border-border"
                            )}
                          >
                            {algorithm.metrics.psnr.toFixed(1)} dB
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
              <Card className="border-border bg-card/50">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-accent border border-border">
                      <Zap className="h-5 w-5" strokeWidth={1.5} />
                    </div>
                    <CardTitle>Detalhes</CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {[
                    { label: "Imagem Original", value: `${originalImage.width}×${originalImage.height}`, icon: FileImage },
                    { label: "Formatos Processados", value: processedImages.length.toString(), icon: Box },
                    { label: "Tempo Total", value: processedImages.length > 0 ? `${processedImages.reduce((sum, alg) => sum + (alg.processingTime || 0), 0).toFixed(2)}s` : "N/A", icon: Clock },
                    { label: "Melhor PSNR", value: processedImages.length > 0 ? `${Math.max(...processedImages.map(a => a.metrics.psnr)).toFixed(2)} dB` : "N/A", icon: TrendingUp },
                  ].map((item, i) => (
                    <div key={i} className="flex items-center justify-between text-sm bg-accent/50 rounded-xl px-4 py-3 border border-border">
                      <span className="flex items-center gap-2 text-muted-foreground">
                        <item.icon className="h-4 w-4" strokeWidth={1.5} />{item.label}
                      </span>
                      <span className="font-medium">{item.value}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Detailed Metrics */}
          <Card className="border-border bg-card/50">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-accent border border-border">
                  <BarChart3 className="h-5 w-5" strokeWidth={1.5} />
                </div>
                <div>
                  <CardTitle>Análise de Métricas</CardTitle>
                  <CardDescription>Comparação completa de qualidade e compressão</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="quality">
                <TabsList className="bg-accent/50 border border-border rounded-xl p-1">
                  <TabsTrigger value="quality" className="rounded-lg data-[state=active]:bg-background">Qualidade</TabsTrigger>
                  <TabsTrigger value="compression" className="rounded-lg data-[state=active]:bg-background">Compressão</TabsTrigger>
                  <TabsTrigger value="performance" className="rounded-lg data-[state=active]:bg-background">Performance</TabsTrigger>
                </TabsList>
                <TabsContent value="quality" className="mt-6">
                  <div className="grid gap-6 md:grid-cols-2">
                    <div className="space-y-4">
                      <h4 className="font-semibold flex items-center gap-2">
                        <TrendingUp className="h-4 w-4" strokeWidth={1.5} />
                        PSNR (Peak Signal-to-Noise Ratio)
                      </h4>
                      {processedImages.map((alg, i) => (
                        <div key={i} className="flex justify-between text-sm bg-accent/50 rounded-xl px-4 py-3 border border-border">
                          <span>{alg.name}</span>
                          <span className="font-medium font-mono">{alg.metrics.psnr.toFixed(2)} dB</span>
                        </div>
                      ))}
                    </div>
                    <div className="space-y-4">
                      <h4 className="font-semibold flex items-center gap-2">
                        <TrendingUp className="h-4 w-4" strokeWidth={1.5} />
                        SSIM (Structural Similarity)
                      </h4>
                      {processedImages.map((alg, i) => (
                        <div key={i} className="flex justify-between text-sm bg-accent/50 rounded-xl px-4 py-3 border border-border">
                          <span>{alg.name}</span>
                          <span className="font-medium font-mono">{alg.metrics.ssim.toFixed(4)}</span>
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
                        <div key={i} className="flex justify-between text-sm bg-accent/50 rounded-xl px-4 py-3 border border-border">
                          <span>{alg.name}</span>
                          <span className="font-medium font-mono">{(alg.metrics.compressionRatio * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                    <div className="space-y-4">
                      <h4 className="font-semibold">Tamanho do Arquivo</h4>
                      {processedImages.map((alg, i) => (
                        <div key={i} className="flex justify-between text-sm bg-accent/50 rounded-xl px-4 py-3 border border-border">
                          <span>{alg.name}</span>
                          <span className="font-medium font-mono">{alg.metrics.fileSize} KB</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </TabsContent>
                <TabsContent value="performance" className="mt-6">
                  <div className="text-center py-16">
                    <div className="p-4 rounded-2xl bg-accent border border-border inline-block mb-4">
                      <BarChart3 className="h-8 w-8 text-muted-foreground" strokeWidth={1.5} />
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
