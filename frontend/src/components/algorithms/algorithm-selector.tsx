"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlgorithmConfig, AlgorithmType, ALGORITHMS, ALGORITHM_CATEGORIES } from "@/lib/algorithms";
import { AlgorithmParams } from "@/lib/types";
import { Settings, Play, Info, Zap } from "lucide-react";
import { motion } from "framer-motion";

interface AlgorithmSelectorProps {
  selectedAlgorithms: { type: AlgorithmType; parameters: AlgorithmParams }[];
  onAlgorithmSelect: (algorithms: { type: AlgorithmType; parameters: AlgorithmParams }[]) => void;
  maxSelections?: number;
}

interface AlgorithmCardProps {
  algorithm: AlgorithmConfig;
  isSelected: boolean;
  parameters: AlgorithmParams;
  onSelect: () => void;
  onParametersChange: (params: AlgorithmParams) => void;
}

function AlgorithmCard({
  algorithm,
  isSelected,
  parameters,
  onSelect,
  onParametersChange
}: AlgorithmCardProps) {
  const [showConfig, setShowConfig] = useState(false);

  const complexityColors = {
    low: "text-green-600 bg-green-100",
    medium: "text-yellow-600 bg-yellow-100",
    high: "text-red-600 bg-red-100",
  };

  const handleParameterChange = (paramName: string, value: number | string | boolean) => {
    const newParams = { ...parameters, [paramName]: value };
    onParametersChange(newParams);
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <Card className={`cursor-pointer transition-all ${isSelected ? 'ring-2 ring-primary' : ''}`}>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-3">
              <div className={`rounded-lg p-2 ${algorithm.color} text-white`}>
                <span className="text-lg">{algorithm.icon}</span>
              </div>
              <div>
                <CardTitle className="text-lg">{algorithm.name}</CardTitle>
                <CardDescription className="text-sm">
                  {algorithm.description}
                </CardDescription>
              </div>
            </div>
            <Checkbox
              checked={isSelected}
              onCheckedChange={onSelect}
            />
          </div>
        </CardHeader>

        <CardContent className="pt-0">
          <div className="flex items-center justify-between mb-4">
            <Badge className={complexityColors[algorithm.complexity]}>
              {algorithm.complexity === 'low' ? 'Baixa' :
               algorithm.complexity === 'medium' ? 'Média' : 'Alta'} Complexidade
            </Badge>

            <div className="flex items-center space-x-1">
              {algorithm.parameters.length > 0 && (
                <Dialog open={showConfig} onOpenChange={setShowConfig}>
                  <DialogTrigger asChild>
                    <Button variant="ghost" size="sm">
                      <Settings className="h-4 w-4" />
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-md">
                    <DialogHeader>
                      <DialogTitle>Configurar {algorithm.name}</DialogTitle>
                      <DialogDescription>
                        Ajuste os parâmetros do algoritmo
                      </DialogDescription>
                    </DialogHeader>

                    <div className="space-y-4 py-4">
                      {algorithm.parameters.map((param) => (
                        <div key={param.name} className="space-y-2">
                          <Label htmlFor={param.name} className="text-sm font-medium">
                            {param.label}
                          </Label>

                          {param.type === 'slider' && (
                            <div className="space-y-2">
                              <Slider
                                id={param.name}
                                min={param.min}
                                max={param.max}
                                step={param.step}
                                value={[parameters[param.name] || param.default] as [number]}
                                onValueChange={(value: [number]) => handleParameterChange(param.name, value[0])}
                                className="w-full"
                              />
                              <div className="flex justify-between text-xs text-muted-foreground">
                                <span>{param.min}</span>
                                <span>{parameters[param.name] || param.default}</span>
                                <span>{param.max}</span>
                              </div>
                            </div>
                          )}

                          {param.type === 'number' && (
                            <Input
                              id={param.name}
                              type="number"
                              min={param.min}
                              max={param.max}
                              value={parameters[param.name] || param.default as any}
                              onChange={(e) => handleParameterChange(param.name, parseFloat(e.target.value))}
                            />
                          )}

                          {param.type === 'select' && (
                            <select
                              id={param.name}
                              value={parameters[param.name] || param.default as any}
                              onChange={(e) => handleParameterChange(param.name, e.target.value)}
                              className="w-full p-2 border rounded-md"
                            >
                              {param.options?.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                          )}

                          {param.description && (
                            <p className="text-xs text-muted-foreground">
                              {param.description}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </DialogContent>
                </Dialog>
              )}

              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="ghost" size="sm">
                    <Info className="h-4 w-4" />
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Sobre {algorithm.name}</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4">
                    <p className="text-sm text-muted-foreground">
                      {algorithm.description}
                    </p>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <strong>Categoria:</strong><br />
                        {ALGORITHM_CATEGORIES[algorithm.category].name}
                      </div>
                      <div>
                        <strong>Complexidade:</strong><br />
                        <Badge className={complexityColors[algorithm.complexity]}>
                          {algorithm.complexity === 'low' ? 'Baixa' :
                           algorithm.complexity === 'medium' ? 'Média' : 'Alta'}
                        </Badge>
                      </div>
                    </div>

                    {algorithm.parameters.length > 0 && (
                      <div>
                        <strong>Parâmetros ({algorithm.parameters.length}):</strong>
                        <ul className="mt-2 space-y-1 text-sm">
                          {algorithm.parameters.map((param) => (
                            <li key={param.name} className="flex justify-between">
                              <span>{param.label}</span>
                              <span className="text-muted-foreground">
                                {param.type === 'slider' ? 'range' :
                                 param.type === 'select' ? 'opções' : 'valor'}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </div>

          {isSelected && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="flex items-center gap-2 text-sm text-green-600"
            >
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              Selecionado para processamento
            </motion.div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

export function AlgorithmSelector({
  selectedAlgorithms,
  onAlgorithmSelect,
  maxSelections = 5
}: AlgorithmSelectorProps) {
  const [activeCategory, setActiveCategory] = useState<string>('decomposition');

  const handleAlgorithmToggle = (algorithmType: AlgorithmType) => {
    const isSelected = selectedAlgorithms.some(alg => alg.type === algorithmType);

    if (isSelected) {
      // Remove algorithm
      const newSelection = selectedAlgorithms.filter(alg => alg.type !== algorithmType);
      onAlgorithmSelect(newSelection);
    } else {
      // Add algorithm with default parameters
      if (selectedAlgorithms.length >= maxSelections) {
        return; // Max selections reached
      }

      const algorithm = ALGORITHMS[algorithmType];
      const defaultParams = algorithm.parameters.reduce((acc, param) => {
        acc[param.name] = param.default;
        return acc;
      }, {} as AlgorithmParams);

      const newSelection = [
        ...selectedAlgorithms,
        { type: algorithmType, parameters: defaultParams }
      ];
      onAlgorithmSelect(newSelection);
    }
  };

  const handleParametersChange = (algorithmType: AlgorithmType, params: AlgorithmParams) => {
    const newSelection = selectedAlgorithms.map(alg => {
      if (alg.type === algorithmType) {
        return { ...alg, parameters: params };
      }
      return alg;
    });
    onAlgorithmSelect(newSelection);
  };

  const getAlgorithmsByCategory = (category: string) => {
    return Object.values(ALGORITHMS).filter(alg => alg.category === category);
  };

  const totalSelected = selectedAlgorithms.length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Selecionar Algoritmos</h2>
          <p className="text-muted-foreground">
            Escolha os algoritmos de processamento de imagens
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Badge variant="outline">
            {totalSelected}/{maxSelections} selecionados
          </Badge>
          {totalSelected > 0 && (
            <Button className="flex items-center gap-2">
              <Play className="h-4 w-4" />
              Processar ({totalSelected})
            </Button>
          )}
        </div>
      </div>

      {totalSelected >= maxSelections && (
        <Alert>
          <AlertDescription>
            Você atingiu o limite máximo de {maxSelections} algoritmos selecionados.
            Desmarque alguns para selecionar outros.
          </AlertDescription>
        </Alert>
      )}

      {/* Algorithm Categories */}
      <Tabs value={activeCategory} onValueChange={setActiveCategory}>
        <TabsList className="grid w-full grid-cols-4">
          {Object.entries(ALGORITHM_CATEGORIES).map(([key, category]) => (
            <TabsTrigger key={key} value={key} className="text-sm">
              {category.name}
            </TabsTrigger>
          ))}
        </TabsList>

        {Object.entries(ALGORITHM_CATEGORIES).map(([categoryKey, category]) => (
          <TabsContent key={categoryKey} value={categoryKey} className="space-y-4">
            <div className={`p-4 rounded-lg ${category.color}`}>
              <h3 className="font-medium">{category.name}</h3>
              <p className="text-sm opacity-80">{category.description}</p>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {getAlgorithmsByCategory(categoryKey).map((algorithm) => {
                const selectedAlg = selectedAlgorithms.find(alg => alg.type === algorithm.type);
                const isSelected = !!selectedAlg;

                return (
                  <AlgorithmCard
                    key={algorithm.type}
                    algorithm={algorithm}
                    isSelected={isSelected}
                    parameters={selectedAlg?.parameters || {}}
                    onSelect={() => handleAlgorithmToggle(algorithm.type)}
                    onParametersChange={(params) => handleParametersChange(algorithm.type, params)}
                  />
                );
              })}
            </div>
          </TabsContent>
        ))}
      </Tabs>

      {/* Selected Algorithms Summary */}
      {totalSelected > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Algoritmos Selecionados</CardTitle>
            <CardDescription>
              Resumo dos algoritmos que serão executados
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {selectedAlgorithms.map((alg) => {
                const config = ALGORITHMS[alg.type];
                return (
                  <div key={alg.type} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className={`rounded-lg p-2 ${config.color} text-white`}>
                        <span className="text-sm">{config.icon}</span>
                      </div>
                      <div>
                        <h4 className="font-medium">{config.name}</h4>
                        <p className="text-sm text-muted-foreground">
                          {config.parameters.length} parâmetro(s) configurado(s)
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">
                        {config.complexity === 'low' ? 'Rápido' :
                         config.complexity === 'medium' ? 'Moderado' : 'Lento'}
                      </Badge>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleAlgorithmToggle(alg.type)}
                      >
                        Remover
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>

            <Separator className="my-4" />

            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Tempo estimado: ~{(totalSelected * 2.3).toFixed(1)}s por imagem
              </div>
              <Button className="flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Iniciar Processamento
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
