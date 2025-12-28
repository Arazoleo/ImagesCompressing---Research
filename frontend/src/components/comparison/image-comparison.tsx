"use client";

import { useState, useRef, useCallback } from "react";
import Image from "next/image";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ImageFile } from "@/lib/types";
import { Move, RotateCcw, Image as ImageIcon } from "lucide-react";
import { motion } from "framer-motion";

interface ImageComparisonProps {
  images: ImageFile[];
  zoomLevel: number;
  syncZoom: boolean;
}

export function ImageComparison({ images, zoomLevel, syncZoom }: ImageComparisonProps) {
  const [panPosition, setPanPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({
      x: e.clientX - panPosition.x,
      y: e.clientY - panPosition.y
    });
  }, [panPosition]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;

    setPanPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const resetPan = () => {
    setPanPosition({ x: 0, y: 0 });
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
        <div className="flex items-center gap-4">
          <Button variant="outline" size="sm" onClick={resetPan}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset Pan
          </Button>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Move className="h-4 w-4" />
            Clique e arraste para mover
          </div>
        </div>

        <div className="text-sm text-muted-foreground">
          Zoom: {Math.round(zoomLevel * 100)}% |
          Pan: ({Math.round(panPosition.x)}, {Math.round(panPosition.y)})
        </div>
      </div>

      {/* Comparison Grid */}
      <div
        ref={containerRef}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6 overflow-hidden"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      >
        {images.map((image, index) => (
          <motion.div
            key={image.id}
            initial={{ opacity: 0, x: index === 0 ? -20 : 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.2 }}
            className="relative"
          >
            <Card className="overflow-hidden">
              <div className="relative aspect-square overflow-hidden bg-muted">
                <div
                  className="relative w-full h-full"
                  style={{
                    transform: `scale(${zoomLevel}) translate(${panPosition.x / zoomLevel}px, ${panPosition.y / zoomLevel}px)`,
                    transformOrigin: 'center',
                    transition: isDragging ? 'none' : 'transform 0.1s ease-out'
                  }}
                >
                  <Image
                    src={image.url}
                    alt={image.name || `Image ${index + 1}`}
                    fill
                    className="object-contain"
                    sizes="(max-width: 768px) 100vw, 50vw"
                    draggable={false}
                  />
                </div>

                {/* Zoom indicator overlay */}
                {zoomLevel > 1 && (
                  <div className="absolute top-4 right-4 bg-black/80 text-white px-2 py-1 rounded text-xs">
                    {Math.round(zoomLevel * 100)}%
                  </div>
                )}
              </div>

              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold">
                    {image.name || `Imagem ${index + 1}`}
                  </h3>
                  <Badge variant={index === 0 ? "default" : "secondary"}>
                    {index === 0 ? "Original" : "Processada"}
                  </Badge>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Dimensões:</span>
                    <div className="font-medium">
                      {image.width}×{image.height}
                    </div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Tamanho:</span>
                    <div className="font-medium">
                      {(image.size / 1024).toFixed(1)} KB
                    </div>
                  </div>
                </div>

                {/* Metrics for processed images */}
                {(image as any).metrics && (
                  <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                    <h4 className="font-medium text-sm mb-2">Métricas</h4>
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div>
                        <span className="text-muted-foreground">PSNR:</span>
                        <div className="font-medium">
                          {(image as any).metrics.psnr}dB
                        </div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">SSIM:</span>
                        <div className="font-medium">
                          {(image as any).metrics.ssim}
                        </div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Ratio:</span>
                        <div className="font-medium">
                          {(image as any).metrics.compressionRatio}x
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        ))}

        {/* Fill empty slot if only one image */}
        {images.length === 1 && (
          <Card className="flex items-center justify-center aspect-square">
            <CardContent className="text-center text-muted-foreground">
              <div className="h-12 w-12 mx-auto mb-4 opacity-50 flex items-center justify-center">
                <ImageIcon className="h-12 w-12" />
              </div>
              <p>Selecione uma segunda imagem para comparar</p>
            </CardContent>
          </Card>
        )}
      </div>

    </div>
  );
}
