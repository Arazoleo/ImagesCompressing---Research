"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlgorithmSelector } from "@/components/algorithms/algorithm-selector";
import { ImageUpload } from "@/components/image/image-upload";
import { ImageFile, AlgorithmType } from "@/lib/types";
import { useProcessImages, useProcessingStatus, useProcessingWebSocket } from "@/hooks/use-api";
import { Play, Image as ImageIcon, AlertTriangle, Cpu, Sparkles, Activity, Upload } from "lucide-react";
import { motion, AnimatePresence, Variants } from "framer-motion";
import { cn } from "@/lib/utils";

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 100, damping: 15 } },
};

export default function ProcessPage() {
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<{ type: AlgorithmType; parameters: any }[]>([]);
  const [uploadedImages, setUploadedImages] = useState<ImageFile[]>([]);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const processMutation = useProcessImages();
  const { data: jobStatus } = useProcessingStatus(currentJobId || '', !!currentJobId);
  const { progress: wsProgress, isConnected: wsConnected } = useProcessingWebSocket(currentJobId, !!currentJobId);

  const handleStartProcessing = async () => {
    if (selectedAlgorithms.length === 0 || uploadedImages.length === 0) {
      setError("Selecione pelo menos um algoritmo e tenha imagens carregadas");
      return;
    }
    setError(null);
    try {
      const result = await processMutation.mutateAsync({ imageIds: uploadedImages.map(img => img.id), algorithms: selectedAlgorithms });
      setCurrentJobId(result.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro ao iniciar processamento");
    }
  };

  const totalTasks = jobStatus 
    ? (jobStatus.total_images || 0) * (jobStatus.total_algorithms || 0)
    : selectedAlgorithms.length * uploadedImages.length;
  const completedTasks = jobStatus?.results?.filter((r: any) => r.status === 'completed').length || 0;
  const progress = jobStatus?.progress ?? (totalTasks > 0 ? (completedTasks / totalTasks) * 100 : 0);
  const isProcessing = processMutation.isPending || (jobStatus && jobStatus.status === 'running');

  return (
    <motion.div className="flex-1 space-y-8 p-6 md:p-8" variants={containerVariants} initial="hidden" animate="visible">
      <motion.div variants={itemVariants}>
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Processar Imagens</h1>
        <p className="text-muted-foreground mt-1">Configure e execute algoritmos de processamento</p>
      </motion.div>

      <AnimatePresence>
        {isProcessing && (
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}>
            <Card className="relative overflow-hidden border-border/50 bg-gradient-to-br from-blue-500/5 via-cyan-500/5 to-transparent">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-4">
                    <div className="relative w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shadow-lg">
                      <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: "linear" }}>
                        <Cpu className="h-6 w-6 text-white" />
                      </motion.div>
                    </div>
                    <div>
                      <h3 className="font-semibold text-lg">Processando...</h3>
                      <p className="text-sm text-muted-foreground">{completedTasks} de {totalTasks} tarefas</p>
                    </div>
                  </div>
                  <Badge className="bg-blue-500/10 text-blue-600 border-0 text-lg px-4 py-1">
                    {Math.round(progress)}%
                  </Badge>
                </div>
                <Progress value={progress} className="h-3" />
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
            <Alert variant="destructive"><AlertTriangle className="h-4 w-4" /><AlertDescription>{error}</AlertDescription></Alert>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Image Upload Section */}
      <motion.div variants={itemVariants}>
        <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center shadow-lg">
                <Upload className="h-5 w-5 text-white" />
              </div>
              <div>
                <CardTitle>1. Carregar Imagens</CardTitle>
                <CardDescription>Selecione as imagens que deseja processar</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <ImageUpload onImagesUploaded={setUploadedImages} maxFiles={10} />
          </CardContent>
        </Card>
      </motion.div>

      <div className="grid gap-6 lg:grid-cols-3">
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <div className="mb-6">
            <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle>2. Selecionar Algoritmos</CardTitle>
                <CardDescription>Escolha os algoritmos de processamento a serem aplicados</CardDescription>
              </CardHeader>
            </Card>
          </div>
          <AlgorithmSelector selectedAlgorithms={selectedAlgorithms} onAlgorithmSelect={setSelectedAlgorithms} maxSelections={5} />
        </motion.div>

        <motion.div variants={itemVariants} className="space-y-6">
          <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center shadow-lg">
                  <ImageIcon className="h-5 w-5 text-white" />
                </div>
                <CardTitle>Imagens Carregadas</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {uploadedImages.map((image, index) => (
                  <motion.div key={image.id} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * 0.1 }} className="flex items-center gap-3 p-3 bg-muted/30 rounded-xl">
                    <div className="w-12 h-12 rounded-lg bg-muted flex items-center justify-center"><ImageIcon className="h-5 w-5 text-muted-foreground" /></div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{image.name}</p>
                      <p className="text-xs text-muted-foreground">{image.width}×{image.height} px</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center shadow-lg">
                  <Sparkles className="h-5 w-5 text-primary-foreground" />
                </div>
                <div>
                  <CardTitle>3. Iniciar Processamento</CardTitle>
                  <CardDescription>Execute os algoritmos selecionados nas imagens carregadas</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-muted/30 rounded-xl">
                  <div className="text-3xl font-bold text-primary">{selectedAlgorithms.length}</div>
                  <div className="text-xs text-muted-foreground">Algoritmos</div>
                </div>
                <div className="text-center p-4 bg-muted/30 rounded-xl">
                  <div className="text-3xl font-bold text-primary">{uploadedImages.length}</div>
                  <div className="text-xs text-muted-foreground">Imagens</div>
                </div>
              </div>
              <Button onClick={handleStartProcessing} disabled={selectedAlgorithms.length === 0 || uploadedImages.length === 0 || isProcessing} className="w-full h-12" size="lg">
                {isProcessing ? <><Cpu className="h-5 w-5 mr-2 animate-spin" />Processando...</> : <><Play className="h-5 w-5 mr-2" />Iniciar Processamento</>}
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  );
}
