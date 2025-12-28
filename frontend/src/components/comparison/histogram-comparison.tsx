"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ImageFile } from "@/lib/types";
import { BarChart3, TrendingUp } from "lucide-react";
import { motion } from "framer-motion";

interface HistogramComparisonProps {
  images: ImageFile[];
}

export function HistogramComparison({ images }: HistogramComparisonProps) {
  const [histogramData, setHistogramData] = useState<any[]>([]);

  // Mock histogram data - in real implementation, this would be calculated from actual image data
  useEffect(() => {
    const mockData = [
      {
        label: "0-64",
        original: 1200,
        processed: 1150
      },
      {
        label: "64-128",
        original: 1800,
        processed: 1750
      },
      {
        label: "128-192",
        original: 2200,
        processed: 2100
      },
      {
        label: "192-255",
        original: 1600,
        processed: 1550
      }
    ];
    setHistogramData(mockData);
  }, [images]);

  const maxValue = Math.max(
    ...histogramData.flatMap(d => [d.original, d.processed])
  );

  if (images.length < 2) {
    return (
      <Card className="p-8 text-center">
        <div className="text-muted-foreground mb-4">
          <BarChart3 className="h-12 w-12 mx-auto" />
        </div>
        <h3 className="text-lg font-semibold mb-2">Selecione 2 Imagens</h3>
        <p className="text-muted-foreground">
          A comparação de histogramas requer exatamente 2 imagens.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="intensity" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="intensity">Intensidade</TabsTrigger>
          <TabsTrigger value="rgb">RGB</TabsTrigger>
          <TabsTrigger value="statistics">Estatísticas</TabsTrigger>
        </TabsList>

        <TabsContent value="intensity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Histograma de Intensidade</CardTitle>
              <CardDescription>
                Distribuição dos valores de intensidade (0-255) das imagens
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Legend */}
                <div className="flex items-center gap-6">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-blue-500 rounded"></div>
                    <span className="text-sm">{images[0]?.name || "Imagem 1"}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-green-500 rounded"></div>
                    <span className="text-sm">{images[1]?.name || "Imagem 2"}</span>
                  </div>
                </div>

                {/* Histogram Bars */}
                <div className="space-y-2">
                  {histogramData.map((bin, index) => (
                    <motion.div
                      key={bin.label}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex items-center gap-4"
                    >
                      <div className="w-16 text-sm font-medium text-right">
                        {bin.label}
                      </div>

                      <div className="flex-1 flex gap-2">
                        {/* Original image bar */}
                        <div className="flex-1 relative">
                          <div
                            className="bg-blue-500 h-8 rounded transition-all duration-500"
                            style={{
                              width: `${(bin.original / maxValue) * 100}%`
                            }}
                          ></div>
                          <div className="absolute -top-6 left-0 text-xs text-blue-600 font-medium">
                            {bin.original.toLocaleString()}
                          </div>
                        </div>

                        {/* Processed image bar */}
                        <div className="flex-1 relative">
                          <div
                            className="bg-green-500 h-8 rounded transition-all duration-500"
                            style={{
                              width: `${(bin.processed / maxValue) * 100}%`
                            }}
                          ></div>
                          <div className="absolute -top-6 right-0 text-xs text-green-600 font-medium">
                            {bin.processed.toLocaleString()}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="rgb" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {["Red", "Green", "Blue"].map((channel, channelIndex) => (
              <Card key={channel}>
                <CardHeader>
                  <CardTitle className="text-lg">{channel} Channel</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {[0, 1, 2, 3].map((binIndex) => {
                      const bin = histogramData[binIndex];
                      const originalValue = bin ? bin.original * (0.8 + Math.random() * 0.4) : 0;
                      const processedValue = bin ? bin.processed * (0.8 + Math.random() * 0.4) : 0;
                      const channelMax = Math.max(originalValue, processedValue);

                      return (
                        <div key={binIndex} className="flex items-center gap-2">
                          <div className="w-8 text-xs text-muted-foreground">
                            {bin?.label}
                          </div>
                          <div className="flex-1 flex gap-1">
                            <div
                              className={`h-6 rounded opacity-80`}
                              style={{
                                width: `${(originalValue / channelMax) * 100}%`,
                                backgroundColor: channel === "Red" ? "#ef4444" :
                                               channel === "Green" ? "#22c55e" : "#3b82f6"
                              }}
                            ></div>
                            <div
                              className={`h-6 rounded opacity-60`}
                              style={{
                                width: `${(processedValue / channelMax) * 100}%`,
                                backgroundColor: channel === "Red" ? "#fca5a5" :
                                               channel === "Green" ? "#86efac" : "#93c5fd"
                              }}
                            ></div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="statistics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {images.map((image, index) => (
              <Card key={image.id}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    {image.name || `Imagem ${index + 1}`}
                    <Badge variant={index === 0 ? "default" : "secondary"}>
                      {index === 0 ? "Original" : "Processada"}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-3 bg-muted/50 rounded-lg">
                        <div className="text-lg font-bold">128.5</div>
                        <div className="text-xs text-muted-foreground">Média</div>
                      </div>
                      <div className="text-center p-3 bg-muted/50 rounded-lg">
                        <div className="text-lg font-bold">45.2</div>
                        <div className="text-xs text-muted-foreground">Desvio</div>
                      </div>
                      <div className="text-center p-3 bg-muted/50 rounded-lg">
                        <div className="text-lg font-bold">67</div>
                        <div className="text-xs text-muted-foreground">Mediana</div>
                      </div>
                      <div className="text-center p-3 bg-muted/50 rounded-lg">
                        <div className="text-lg font-bold">89.3%</div>
                        <div className="text-xs text-muted-foreground">Entropia</div>
                      </div>
                    </div>

                    <div className="pt-4 border-t">
                      <h4 className="font-medium text-sm mb-2">Distribuição por Quartil</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Q1 (25%):</span>
                          <span className="font-medium">64.2</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Q2 (50%):</span>
                          <span className="font-medium">128.5</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Q3 (75%):</span>
                          <span className="font-medium">192.8</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
