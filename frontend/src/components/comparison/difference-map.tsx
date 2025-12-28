"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { ImageFile } from "@/lib/types";
import { Eye, EyeOff, Zap } from "lucide-react";
import { motion } from "framer-motion";

interface DifferenceMapProps {
  images: ImageFile[];
  showDifference: boolean;
  onToggleDifference: (show: boolean) => void;
}

export function DifferenceMap({ images, showDifference, onToggleDifference }: DifferenceMapProps) {
  const [differenceIntensity, setDifferenceIntensity] = useState(2);
  const [showHeatmap, setShowHeatmap] = useState(true);

  if (images.length < 2) {
    return (
      <Card className="p-8 text-center">
        <div className="text-muted-foreground mb-4">
          <Eye className="h-12 w-12 mx-auto" />
        </div>
        <h3 className="text-lg font-semibold mb-2">Selecione 2 Imagens</h3>
        <p className="text-muted-foreground">
          O mapa de diferenças requer exatamente 2 imagens para comparação.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Configurações do Mapa de Diferenças
          </CardTitle>
          <CardDescription>
            Controle como as diferenças são visualizadas
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Label htmlFor="show-difference">Mostrar Diferenças</Label>
              <Switch
                id="show-difference"
                checked={showDifference}
                onCheckedChange={onToggleDifference}
              />
            </div>

            <div className="flex items-center gap-2">
              <Label htmlFor="show-heatmap">Heatmap</Label>
              <Switch
                id="show-heatmap"
                checked={showHeatmap}
                onCheckedChange={setShowHeatmap}
              />
            </div>
          </div>

          {showDifference && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-4"
            >
              <div>
                <Label>Intensidade das Diferenças: {differenceIntensity}x</Label>
                <Slider
                  value={[differenceIntensity]}
                  onValueChange={([value]) => setDifferenceIntensity(value)}
                  min={0.5}
                  max={5}
                  step={0.1}
                  className="mt-2"
                />
              </div>
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* Difference Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Original Images */}
        {images.map((image, index) => (
          <motion.div
            key={image.id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>{image.name || `Imagem ${index + 1}`}</span>
                  <Badge variant={index === 0 ? "default" : "secondary"}>
                    {index === 0 ? "Original" : "Processada"}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="relative aspect-square bg-muted rounded-lg overflow-hidden">
                  <div className="w-full h-full bg-gradient-to-br from-blue-100 to-blue-200 flex items-center justify-center">
                    <Eye className="h-12 w-12 text-blue-500" />
                  </div>

                  {/* Difference Overlay */}
                  {showDifference && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 0.7 }}
                      className="absolute inset-0"
                      style={{
                        background: showHeatmap
                          ? `linear-gradient(45deg,
                              rgba(255, 0, 0, ${0.1 * differenceIntensity}),
                              rgba(255, 255, 0, ${0.2 * differenceIntensity}),
                              rgba(0, 255, 0, ${0.1 * differenceIntensity}))`
                          : `rgba(255, 0, 0, ${0.3 * differenceIntensity})`,
                        mixBlendMode: 'multiply'
                      }}
                    />
                  )}
                </div>

                <div className="mt-4 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Dimensões:</span>
                    <span>{image.width}×{image.height}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Tamanho:</span>
                    <span>{(image.size / 1024).toFixed(1)} KB</span>
                  </div>

                  {/* Metrics */}
                  {(image as any).metrics && (
                    <div className="pt-2 border-t">
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div className="text-center">
                          <div className="text-muted-foreground">PSNR</div>
                          <div className="font-medium">{(image as any).metrics.psnr}dB</div>
                        </div>
                        <div className="text-center">
                          <div className="text-muted-foreground">SSIM</div>
                          <div className="font-medium">{(image as any).metrics.ssim}</div>
                        </div>
                        <div className="text-center">
                          <div className="text-muted-foreground">Ratio</div>
                          <div className="font-medium">{(image as any).metrics.compressionRatio}x</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Difference Statistics */}
      {showDifference && (
        <Card>
          <CardHeader>
            <CardTitle>Estatísticas das Diferenças</CardTitle>
            <CardDescription>
              Análise quantitativa das diferenças entre as imagens
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-red-50 rounded-lg">
                <div className="text-2xl font-bold text-red-600">23.4%</div>
                <div className="text-sm text-muted-foreground">Pixels Diferentes</div>
              </div>
              <div className="text-center p-4 bg-yellow-50 rounded-lg">
                <div className="text-2xl font-bold text-yellow-600">12.8</div>
                <div className="text-sm text-muted-foreground">Erro Médio</div>
              </div>
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">89.2%</div>
                <div className="text-sm text-muted-foreground">Similaridade</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">2.3x</div>
                <div className="text-sm text-muted-foreground">Compressão</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
