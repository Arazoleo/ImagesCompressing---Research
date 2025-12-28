"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { ImageFile } from "@/lib/types";
import { useImages } from "@/hooks/use-api";
import { ArrowLeft, ZoomIn, ZoomOut, RotateCcw, Eye, BarChart3, Image as ImageIcon, Layers } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";
import { ImageComparison } from "@/components/comparison/image-comparison";
import { DifferenceMap } from "@/components/comparison/difference-map";
import { HistogramComparison } from "@/components/comparison/histogram-comparison";
import { MetricsVisualization } from "@/components/comparison/metrics-visualization";
import { cn } from "@/lib/utils";

export default function ComparePage() {
  const { data: images, isLoading } = useImages(50, 0);
  const [selectedImages, setSelectedImages] = useState<ImageFile[]>([]);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showDifference, setShowDifference] = useState(false);
  const [syncZoom, setSyncZoom] = useState(true);

  const [processedImages] = useState([
    { id: "processed-1", originalId: "1", algorithm: "svd", url: "/api/placeholder/512/512", metrics: { psnr: 17.88, ssim: 0.85, compressionRatio: 2.1 }, filename: "image_svd_k25.jpg" },
    { id: "processed-2", originalId: "1", algorithm: "pca", url: "/api/placeholder/512/512", metrics: { psnr: 38.57, ssim: 0.92, compressionRatio: 1.8 }, filename: "image_pca_comp20.jpg" },
    { id: "processed-3", originalId: "1", algorithm: "wavelet", url: "/api/placeholder/512/512", metrics: { psnr: 51.66, ssim: 0.96, compressionRatio: 3.2 }, filename: "image_wavelet.jpg" },
    { id: "processed-4", originalId: "1", algorithm: "autoencoder", url: "/api/placeholder/512/512", metrics: { psnr: 1.49, ssim: 0.007, compressionRatio: 8.5 }, filename: "image_ae.png" }
  ]);

  const handleImageSelect = (image: any) => {
    if (selectedImages.find(img => img.id === image.id)) {
      setSelectedImages(selectedImages.filter(img => img.id !== image.id));
    } else if (selectedImages.length < 2) {
      setSelectedImages([...selectedImages, image]);
    }
  };

  const resetView = () => { setZoomLevel(1); setShowDifference(false); };

  return (
    <motion.div className="flex-1 space-y-8 p-6 md:p-8" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <Link href="/"><Button variant="outline" size="sm" className="rounded-xl"><ArrowLeft className="h-4 w-4 mr-2" />Voltar</Button></Link>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Comparação Visual</h1>
            <p className="text-muted-foreground mt-1">Compare algoritmos lado a lado</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Label htmlFor="sync-zoom" className="text-sm">Zoom Sincronizado</Label>
            <Switch id="sync-zoom" checked={syncZoom} onCheckedChange={setSyncZoom} />
          </div>
          <div className="flex items-center gap-2 bg-muted/30 rounded-xl p-1">
            <Button variant="ghost" size="icon" onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.2))} className="rounded-lg h-8 w-8"><ZoomOut className="h-4 w-4" /></Button>
            <span className="text-sm font-medium min-w-[50px] text-center">{Math.round(zoomLevel * 100)}%</span>
            <Button variant="ghost" size="icon" onClick={() => setZoomLevel(Math.min(3, zoomLevel + 0.2))} className="rounded-lg h-8 w-8"><ZoomIn className="h-4 w-4" /></Button>
          </div>
          <Button variant="outline" onClick={resetView} className="rounded-xl"><RotateCcw className="h-4 w-4 mr-2" />Reset</Button>
        </div>
      </div>

      {/* Image Selection */}
      <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center shadow-lg">
              <ImageIcon className="h-5 w-5 text-white" />
            </div>
            <div>
              <CardTitle>Selecionar Imagens</CardTitle>
              <CardDescription>Escolha até 2 imagens para comparar</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {processedImages.map((image, index) => (
              <motion.div
                key={image.id}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ scale: 1.03, y: -2 }}
                className={cn(
                  "relative cursor-pointer rounded-xl overflow-hidden border-2 transition-all duration-300 group",
                  selectedImages.find(img => img.id === image.id)
                    ? "border-primary shadow-lg shadow-primary/20"
                    : "border-border/50 hover:border-primary/50"
                )}
                onClick={() => handleImageSelect(image)}
              >
                <div className="aspect-square bg-gradient-to-br from-muted to-muted/50 flex items-center justify-center">
                  <Layers className="h-8 w-8 text-muted-foreground group-hover:scale-110 transition-transform" />
                </div>
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent text-white p-3">
                  <div className="text-xs font-semibold uppercase">{image.algorithm}</div>
                  <div className="text-[10px] opacity-80">PSNR: {image.metrics.psnr}dB</div>
                </div>
                {selectedImages.find(img => img.id === image.id) && (
                  <Badge className="absolute top-2 right-2 bg-primary">{selectedImages.findIndex(img => img.id === image.id) + 1}</Badge>
                )}
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Comparison */}
      {selectedImages.length >= 1 ? (
        <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center shadow-lg">
                <BarChart3 className="h-5 w-5 text-white" />
              </div>
              <CardTitle>Análise Comparativa</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="side-by-side">
              <TabsList className="grid w-full grid-cols-4 bg-muted/30 rounded-xl p-1">
                <TabsTrigger value="side-by-side" className="rounded-lg">Lado a Lado</TabsTrigger>
                <TabsTrigger value="difference" className="rounded-lg">Diferenças</TabsTrigger>
                <TabsTrigger value="histogram" className="rounded-lg">Histogramas</TabsTrigger>
                <TabsTrigger value="metrics" className="rounded-lg">Métricas</TabsTrigger>
              </TabsList>
              <TabsContent value="side-by-side" className="mt-6"><ImageComparison images={selectedImages} zoomLevel={zoomLevel} syncZoom={syncZoom} /></TabsContent>
              <TabsContent value="difference" className="mt-6"><DifferenceMap images={selectedImages} showDifference={showDifference} onToggleDifference={setShowDifference} /></TabsContent>
              <TabsContent value="histogram" className="mt-6"><HistogramComparison images={selectedImages} /></TabsContent>
              <TabsContent value="metrics" className="mt-6"><MetricsVisualization images={selectedImages} /></TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      ) : (
        <Card className="border-border/50 p-12 text-center bg-card/50">
          <div className="w-20 h-20 rounded-full bg-muted/50 flex items-center justify-center mx-auto mb-4">
            <Eye className="h-10 w-10 text-muted-foreground" />
          </div>
          <h3 className="text-xl font-semibold mb-2">Selecione Imagens para Comparar</h3>
          <p className="text-muted-foreground max-w-md mx-auto">Escolha até 2 imagens acima para visualizar comparações detalhadas</p>
        </Card>
      )}
    </motion.div>
  );
}
