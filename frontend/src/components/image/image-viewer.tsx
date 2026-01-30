"use client";

import { useState, useRef, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Eye,
  BarChart3,
  SplitSquareHorizontal,
  Image as ImageIcon,
  Download,
  Maximize2
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface ImageData {
  url: string;
  name: string;
  width?: number;
  height?: number;
  metrics?: {
    psnr?: number;
    ssim?: number;
    compressionRatio?: number;
    fileSize?: number;
  };
}

interface ImageViewerProps {
  originalImage: ImageData;
  processedImages?: ImageData[];
  className?: string;
}

export function ImageViewer({
  originalImage,
  processedImages = [],
  className = ""
}: ImageViewerProps) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [viewMode, setViewMode] = useState<'single' | 'compare' | 'difference'>('single');
  const [selectedProcessedImage, setSelectedProcessedImage] = useState(0);
  const [showDifference, setShowDifference] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  }, [zoom, pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.2, 5));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev / 1.2, 0.1));
  };

  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleZoomChange = (value: [number]) => {
    setZoom(value[0]);
  };

  const formatMetric = (value: number, unit: string = '') => {
    if (value >= 1) {
      return value.toFixed(2) + unit;
    }
    return value.toFixed(4) + unit;
  };

  const currentProcessedImage = processedImages[selectedProcessedImage];

  const renderImage = (imageData: ImageData, showControls = true) => (
    <div className="relative overflow-hidden rounded-xl border border-border bg-accent/30">
      <div
        ref={containerRef}
        className="relative overflow-hidden cursor-move"
        style={{ height: '400px' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <motion.img
          ref={imageRef}
          src={imageData.url}
          alt={imageData.name}
          className="absolute top-0 left-0 select-none max-w-none"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'top left',
          }}
          drag={false}
          onError={(e) => {
            // Fallback for broken images
            const target = e.target as HTMLImageElement;
            target.style.display = 'none';
          }}
        />
      </div>

      {showControls && (
        <div className="absolute top-3 left-3 flex gap-1.5">
          <Button 
            size="sm" 
            variant="secondary" 
            onClick={handleZoomIn}
            className="h-8 w-8 p-0 bg-background/80 backdrop-blur-sm border border-border hover:bg-accent"
          >
            <ZoomIn className="h-4 w-4" strokeWidth={1.5} />
          </Button>
          <Button 
            size="sm" 
            variant="secondary" 
            onClick={handleZoomOut}
            className="h-8 w-8 p-0 bg-background/80 backdrop-blur-sm border border-border hover:bg-accent"
          >
            <ZoomOut className="h-4 w-4" strokeWidth={1.5} />
          </Button>
          <Button 
            size="sm" 
            variant="secondary" 
            onClick={handleReset}
            className="h-8 w-8 p-0 bg-background/80 backdrop-blur-sm border border-border hover:bg-accent"
          >
            <RotateCcw className="h-4 w-4" strokeWidth={1.5} />
          </Button>
        </div>
      )}

      <div className="absolute bottom-3 left-3">
        <Badge variant="secondary" className="text-xs bg-background/80 backdrop-blur-sm border border-border font-mono">
          {zoom.toFixed(1)}x
        </Badge>
      </div>

      {imageData.metrics && (
        <div className="absolute top-3 right-3 bg-background/90 backdrop-blur-sm rounded-lg p-3 text-xs space-y-1.5 border border-border">
          {imageData.metrics.psnr !== undefined && imageData.metrics.psnr > 0 && (
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">PSNR</span>
              <span className="font-mono font-medium">{formatMetric(imageData.metrics.psnr, ' dB')}</span>
            </div>
          )}
          {imageData.metrics.ssim !== undefined && imageData.metrics.ssim > 0 && (
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">SSIM</span>
              <span className="font-mono font-medium">{formatMetric(imageData.metrics.ssim)}</span>
            </div>
          )}
          {imageData.metrics.compressionRatio !== undefined && imageData.metrics.compressionRatio > 0 && (
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Comp.</span>
              <span className="font-mono font-medium">{formatMetric(imageData.metrics.compressionRatio * 100, '%')}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );

  const renderComparisonView = () => (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <h3 className="text-sm font-medium mb-2 text-center text-muted-foreground">Original</h3>
        {renderImage(originalImage)}
      </div>
      <div>
        <h3 className="text-sm font-medium mb-2 text-center flex items-center justify-center gap-2 text-muted-foreground">
          Processada
          {currentProcessedImage && (
            <Badge variant="secondary" className="text-xs bg-foreground text-background">
              {currentProcessedImage.name}
            </Badge>
          )}
        </h3>
        {currentProcessedImage ? renderImage(currentProcessedImage) : (
          <div className="h-[400px] border border-dashed border-border rounded-xl flex items-center justify-center text-muted-foreground bg-accent/20">
            <div className="text-center">
              <ImageIcon className="h-12 w-12 mx-auto mb-2 opacity-30" strokeWidth={1} />
              <p className="text-sm">Selecione uma imagem processada</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderDifferenceView = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Visualização de Diferenças</h3>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowDifference(!showDifference)}
          className="rounded-lg border-border"
        >
          <Eye className="h-4 w-4 mr-2" strokeWidth={1.5} />
          {showDifference ? 'Ocultar' : 'Mostrar'} Diferença
        </Button>
      </div>

      {showDifference ? (
        <div className="relative">
          {renderImage(originalImage, false)}
          {currentProcessedImage && (
            <div className="absolute inset-0 mix-blend-difference rounded-xl overflow-hidden">
              <img
                src={currentProcessedImage.url}
                alt="difference"
                className="w-full h-full object-cover"
              />
            </div>
          )}
        </div>
      ) : (
        renderComparisonView()
      )}
    </div>
  );

  return (
    <div className={cn("space-y-6", className)}>
      {/* Controls */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-4">
          <Tabs value={viewMode} onValueChange={(value: string) => setViewMode(value as 'single' | 'compare' | 'difference')}>
            <TabsList className="bg-accent/50 border border-border rounded-xl p-1">
              <TabsTrigger value="single" className="text-xs rounded-lg data-[state=active]:bg-background">
                <ImageIcon className="h-4 w-4 mr-1.5" strokeWidth={1.5} />
                Única
              </TabsTrigger>
              <TabsTrigger value="compare" className="text-xs rounded-lg data-[state=active]:bg-background">
                <SplitSquareHorizontal className="h-4 w-4 mr-1.5" strokeWidth={1.5} />
                Comparar
              </TabsTrigger>
              <TabsTrigger value="difference" className="text-xs rounded-lg data-[state=active]:bg-background">
                <BarChart3 className="h-4 w-4 mr-1.5" strokeWidth={1.5} />
                Diferença
              </TabsTrigger>
            </TabsList>
          </Tabs>

          {processedImages.length > 0 && viewMode !== 'single' && (
            <select
              value={selectedProcessedImage}
              onChange={(e) => setSelectedProcessedImage(parseInt(e.target.value))}
              className="text-sm border border-border rounded-lg px-3 py-1.5 bg-accent/50 focus:outline-none focus:ring-1 focus:ring-foreground/20"
            >
              {processedImages.map((img, index) => (
                <option key={index} value={index}>
                  {img.name}
                </option>
              ))}
            </select>
          )}
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="rounded-lg border-border hover:bg-accent">
            <Download className="h-4 w-4 mr-1.5" strokeWidth={1.5} />
            Exportar
          </Button>
          <Button variant="outline" size="sm" className="rounded-lg border-border hover:bg-accent h-8 w-8 p-0">
            <Maximize2 className="h-4 w-4" strokeWidth={1.5} />
          </Button>
        </div>
      </div>

      {/* Zoom Controls */}
      <div className="flex items-center gap-4">
        <span className="text-sm font-medium text-muted-foreground">Zoom</span>
        <Slider
          value={[zoom]}
          onValueChange={handleZoomChange}
          min={0.1}
          max={5}
          step={0.1}
          className="flex-1 max-w-xs"
        />
        <span className="text-sm text-muted-foreground min-w-[3rem] font-mono">
          {zoom.toFixed(1)}x
        </span>
        <Button 
          variant="outline" 
          size="sm" 
          onClick={handleReset}
          className="rounded-lg border-border hover:bg-accent"
        >
          Reset
        </Button>
      </div>

      <Separator className="bg-border" />

      {/* Image Display */}
      {viewMode === 'single' && renderImage(originalImage)}

      {viewMode === 'compare' && renderComparisonView()}

      {viewMode === 'difference' && renderDifferenceView()}

      {/* Metrics Summary */}
      {currentProcessedImage?.metrics && (
        <>
          <Separator className="bg-border" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-accent/30 rounded-xl border border-border">
              <div className="text-2xl font-semibold font-mono">
                {formatMetric(currentProcessedImage.metrics.psnr || 0, '')}
              </div>
              <div className="text-xs text-muted-foreground mt-1">PSNR (dB)</div>
            </div>
            <div className="text-center p-4 bg-accent/30 rounded-xl border border-border">
              <div className="text-2xl font-semibold font-mono">
                {formatMetric(currentProcessedImage.metrics.ssim || 0)}
              </div>
              <div className="text-xs text-muted-foreground mt-1">SSIM</div>
            </div>
            <div className="text-center p-4 bg-accent/30 rounded-xl border border-border">
              <div className="text-2xl font-semibold font-mono">
                {formatMetric((currentProcessedImage.metrics.compressionRatio || 0) * 100, '%')}
              </div>
              <div className="text-xs text-muted-foreground mt-1">Compressão</div>
            </div>
            <div className="text-center p-4 bg-accent/30 rounded-xl border border-border">
              <div className="text-2xl font-semibold font-mono">
                {currentProcessedImage.metrics.fileSize ?
                  `${(currentProcessedImage.metrics.fileSize / 1024).toFixed(1)}` :
                  'N/A'
                }
              </div>
              <div className="text-xs text-muted-foreground mt-1">Tamanho (KB)</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
