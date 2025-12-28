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
    <div className="relative overflow-hidden rounded-lg border bg-muted/20">
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
          className="absolute top-0 left-0 select-none"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'top left',
          }}
          drag={false}
        />
      </div>

      {showControls && (
        <div className="absolute top-2 left-2 flex gap-1">
          <Button size="sm" variant="secondary" onClick={handleZoomIn}>
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button size="sm" variant="secondary" onClick={handleZoomOut}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button size="sm" variant="secondary" onClick={handleReset}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      )}

      <div className="absolute bottom-2 left-2">
        <Badge variant="secondary" className="text-xs">
          {zoom.toFixed(1)}x
        </Badge>
      </div>

      {imageData.metrics && (
        <div className="absolute top-2 right-2 bg-background/80 backdrop-blur-sm rounded-lg p-2 text-xs space-y-1">
          {imageData.metrics.psnr && (
            <div>PSNR: {formatMetric(imageData.metrics.psnr, ' dB')}</div>
          )}
          {imageData.metrics.ssim && (
            <div>SSIM: {formatMetric(imageData.metrics.ssim)}</div>
          )}
          {imageData.metrics.compressionRatio && (
            <div>Compressão: {formatMetric(imageData.metrics.compressionRatio * 100, '%')}</div>
          )}
        </div>
      )}
    </div>
  );

  const renderComparisonView = () => (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <h3 className="text-sm font-medium mb-2 text-center">Original</h3>
        {renderImage(originalImage)}
      </div>
      <div>
        <h3 className="text-sm font-medium mb-2 text-center flex items-center justify-center gap-2">
          Processada
          {currentProcessedImage && (
            <Badge variant="outline" className="text-xs">
              {currentProcessedImage.name}
            </Badge>
          )}
        </h3>
        {currentProcessedImage ? renderImage(currentProcessedImage) : (
          <div className="h-[400px] border-2 border-dashed border-muted-foreground/25 rounded-lg flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <ImageIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p>Selecione uma imagem processada</p>
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
        >
          <Eye className="h-4 w-4 mr-2" />
          {showDifference ? 'Ocultar' : 'Mostrar'} Diferença
        </Button>
      </div>

      {showDifference ? (
        <div className="relative">
          {renderImage(originalImage, false)}
          {currentProcessedImage && (
            <div className="absolute inset-0 mix-blend-difference">
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
    <Card className={className}>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Tabs value={viewMode} onValueChange={(value: string) => setViewMode(value as 'single' | 'compare' | 'difference')}>
                <TabsList>
                  <TabsTrigger value="single" className="text-xs">
                    <ImageIcon className="h-4 w-4 mr-1" />
                    Única
                  </TabsTrigger>
                  <TabsTrigger value="compare" className="text-xs">
                    <SplitSquareHorizontal className="h-4 w-4 mr-1" />
                    Comparar
                  </TabsTrigger>
                  <TabsTrigger value="difference" className="text-xs">
                    <BarChart3 className="h-4 w-4 mr-1" />
                    Diferença
                  </TabsTrigger>
                </TabsList>
              </Tabs>

              {processedImages.length > 0 && viewMode !== 'single' && (
                <select
                  value={selectedProcessedImage}
                  onChange={(e) => setSelectedProcessedImage(parseInt(e.target.value))}
                  className="text-sm border rounded px-2 py-1"
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
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-1" />
                Exportar
              </Button>
              <Button variant="outline" size="sm">
                <Maximize2 className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Zoom Controls */}
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium">Zoom:</span>
            <Slider
              value={[zoom]}
              onValueChange={handleZoomChange}
              min={0.1}
              max={5}
              step={0.1}
              className="flex-1 max-w-xs"
            />
            <span className="text-sm text-muted-foreground min-w-[3rem]">
              {zoom.toFixed(1)}x
            </span>
            <Button variant="outline" size="sm" onClick={handleReset}>
              Reset
            </Button>
          </div>

          <Separator />

          {/* Image Display */}
          {viewMode === 'single' && renderImage(originalImage)}

          {viewMode === 'compare' && renderComparisonView()}

          {viewMode === 'difference' && renderDifferenceView()}

          {/* Metrics Summary */}
          {currentProcessedImage?.metrics && (
            <>
              <Separator />
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {formatMetric(currentProcessedImage.metrics.psnr || 0, ' dB')}
                  </div>
                  <div className="text-sm text-muted-foreground">PSNR</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {formatMetric(currentProcessedImage.metrics.ssim || 0)}
                  </div>
                  <div className="text-sm text-muted-foreground">SSIM</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {formatMetric((currentProcessedImage.metrics.compressionRatio || 0) * 100, '%')}
                  </div>
                  <div className="text-sm text-muted-foreground">Compressão</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {currentProcessedImage.metrics.fileSize ?
                      `${(currentProcessedImage.metrics.fileSize / 1024).toFixed(1)} KB` :
                      'N/A'
                    }
                  </div>
                  <div className="text-sm text-muted-foreground">Tamanho</div>
                </div>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
