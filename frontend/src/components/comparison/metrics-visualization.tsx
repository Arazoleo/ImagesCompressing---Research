"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { ImageFile } from "@/lib/types";
import { BarChart3, TrendingUp, Award, Target } from "lucide-react";
import { motion } from "framer-motion";

interface MetricsVisualizationProps {
  images: ImageFile[];
}

export function MetricsVisualization({ images }: MetricsVisualizationProps) {
  const [selectedMetric, setSelectedMetric] = useState<'psnr' | 'ssim' | 'compression'>('psnr');

  // Mock metrics data - in real implementation, this would come from actual processing results
  const metricsData = [
    { algorithm: 'Original', psnr: 100, ssim: 1.0, compression: 1.0 },
    { algorithm: 'SVD', psnr: 17.88, ssim: 0.85, compression: 2.1 },
    { algorithm: 'PCA', psnr: 38.57, ssim: 0.92, compression: 1.8 },
    { algorithm: 'Wavelet', psnr: 51.66, ssim: 0.96, compression: 3.2 },
    { algorithm: 'FFT', psnr: 12.93, ssim: 0.78, compression: 4.5 },
    { algorithm: 'Autoencoder', psnr: 1.49, ssim: 0.007, compression: 8.5 }
  ];

  const getMetricInfo = (metric: string) => {
    switch (metric) {
      case 'psnr':
        return {
          name: 'PSNR (Peak Signal-to-Noise Ratio)',
          description: 'Medida de qualidade da imagem reconstruída. Valores maiores indicam melhor qualidade.',
          unit: 'dB',
          color: 'text-blue-600'
        };
      case 'ssim':
        return {
          name: 'SSIM (Structural Similarity Index)',
          description: 'Medida de similaridade estrutural entre imagens. Valores próximos de 1.0 são melhores.',
          unit: '',
          color: 'text-green-600'
        };
      case 'compression':
        return {
          name: 'Taxa de Compressão',
          description: 'Quanto a imagem foi comprimida. Valores maiores indicam melhor compressão.',
          unit: 'x',
          color: 'text-purple-600'
        };
      default:
        return { name: '', description: '', unit: '', color: '' };
    }
  };

  const metricInfo = getMetricInfo(selectedMetric);
  const maxValue = Math.max(...metricsData.map(d => d[selectedMetric as keyof typeof d] as number));

  return (
    <div className="space-y-6">
      {/* Metric Selector */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Métricas de Qualidade
          </CardTitle>
          <CardDescription>
            Compare o desempenho dos algoritmos usando diferentes métricas
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { key: 'psnr', label: 'PSNR', icon: Target, color: 'bg-blue-500' },
              { key: 'ssim', label: 'SSIM', icon: Award, color: 'bg-green-500' },
              { key: 'compression', label: 'Compressão', icon: TrendingUp, color: 'bg-purple-500' }
            ].map(({ key, label, icon: Icon, color }) => (
              <Button
                key={key}
                variant={selectedMetric === key ? "default" : "outline"}
                className="h-auto p-4 flex flex-col items-center gap-2"
                onClick={() => setSelectedMetric(key as any)}
              >
                <Icon className="h-6 w-6" />
                <span className="font-medium">{label}</span>
              </Button>
            ))}
          </div>

          <div className="mt-6 p-4 bg-muted/50 rounded-lg">
            <h3 className="font-semibold mb-2">{metricInfo.name}</h3>
            <p className="text-sm text-muted-foreground">{metricInfo.description}</p>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Comparison Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Comparação de Algoritmos</CardTitle>
          <CardDescription>
            Desempenho relativo baseado na métrica selecionada
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {metricsData.map((data, index) => {
              const value = data[selectedMetric as keyof typeof data] as number;
              const percentage = (value / maxValue) * 100;
              const isBest = value === maxValue && selectedMetric !== 'compression';

              return (
                <motion.div
                  key={data.algorithm}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="relative"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <span className="font-medium">{data.algorithm}</span>
                      {isBest && (
                        <Badge className="bg-yellow-500 text-yellow-900">
                          <Award className="h-3 w-3 mr-1" />
                          Melhor
                        </Badge>
                      )}
                    </div>
                    <div className={`font-bold ${metricInfo.color}`}>
                      {value}{metricInfo.unit}
                    </div>
                  </div>

                  <Progress
                    value={percentage}
                    className="h-3"
                  />

                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>0</span>
                    <span>{maxValue.toFixed(1)}{metricInfo.unit}</span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Quality vs Compression Trade-off */}
      <Card>
        <CardHeader>
          <CardTitle>Qualidade vs Compressão</CardTitle>
          <CardDescription>
            Equilíbrio entre qualidade de imagem e taxa de compressão
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Scatter plot mockup */}
            <div className="relative h-64 bg-muted/30 rounded-lg border-2 border-dashed border-muted-foreground/20 flex items-center justify-center">
              <div className="text-center text-muted-foreground">
                <Target className="h-12 w-12 mx-auto mb-4" />
                <p className="text-sm">
                  Gráfico interativo: Qualidade vs Taxa de Compressão
                </p>
                <p className="text-xs mt-2">
                  Cada ponto representa um algoritmo
                </p>
              </div>

              {/* Mock data points */}
              <div className="absolute inset-4">
                {metricsData.slice(1).map((data, index) => {
                  const x = (data.compression / 10) * 100; // Normalize compression
                  const y = (data.psnr / 60) * 100; // Normalize PSNR

                  return (
                    <motion.div
                      key={data.algorithm}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: index * 0.2 }}
                      className="absolute w-3 h-3 bg-primary rounded-full border-2 border-white shadow-lg"
                      style={{
                        left: `${Math.max(5, Math.min(95, x))}%`,
                        bottom: `${Math.max(5, Math.min(95, y))}%`,
                        transform: 'translate(-50%, 50%)'
                      }}
                      title={`${data.algorithm}: PSNR ${data.psnr}dB, Compressão ${data.compression}x`}
                    />
                  );
                })}
              </div>
            </div>

            {/* Trade-off analysis */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-medium text-green-800 mb-2">Melhor Qualidade</h4>
                <p className="text-sm text-green-700">
                  Wavelet oferece o melhor equilíbrio entre qualidade e compressão,
                  com PSNR de 51.66dB e compressão 3.2x.
                </p>
              </div>

              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-medium text-blue-800 mb-2">Melhor Compressão</h4>
                <p className="text-sm text-blue-700">
                  Autoencoder alcança a maior taxa de compressão (8.5x),
                  mas com qualidade significativamente reduzida.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Metrics Table */}
      <Card>
        <CardHeader>
          <CardTitle>Tabela Comparativa Detalhada</CardTitle>
          <CardDescription>
            Valores exatos de todas as métricas para análise precisa
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Algoritmo</th>
                  <th className="text-center py-2">PSNR (dB)</th>
                  <th className="text-center py-2">SSIM</th>
                  <th className="text-center py-2">Compressão</th>
                  <th className="text-center py-2">Eficiência</th>
                </tr>
              </thead>
              <tbody>
                {metricsData.map((data, index) => {
                  const efficiency = (data.psnr / Math.log(data.compression + 1)) * data.ssim;
                  return (
                    <motion.tr
                      key={data.algorithm}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: index * 0.05 }}
                      className="border-b hover:bg-muted/50"
                    >
                      <td className="py-3 font-medium">{data.algorithm}</td>
                      <td className="text-center py-3">
                        <span className={data.psnr > 40 ? 'text-green-600 font-medium' : data.psnr > 20 ? 'text-yellow-600' : 'text-red-600'}>
                          {data.psnr}
                        </span>
                      </td>
                      <td className="text-center py-3">
                        <span className={data.ssim > 0.9 ? 'text-green-600 font-medium' : data.ssim > 0.8 ? 'text-yellow-600' : 'text-red-600'}>
                          {data.ssim}
                        </span>
                      </td>
                      <td className="text-center py-3">
                        <span className={data.compression > 3 ? 'text-green-600 font-medium' : data.compression > 2 ? 'text-yellow-600' : 'text-gray-600'}>
                          {data.compression}x
                        </span>
                      </td>
                      <td className="text-center py-3">
                        <span className="font-medium">{efficiency.toFixed(1)}</span>
                      </td>
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
