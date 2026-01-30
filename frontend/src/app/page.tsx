"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Upload,
  Cpu,
  BarChart3,
  GitCompare,
  FileImage,
  TrendingUp,
  Clock,
  ArrowUpRight,
  CheckCircle,
  XCircle,
  Activity,
  Layers,
  Box,
  Zap,
} from "lucide-react";
import Link from "next/link";
import { motion, Variants } from "framer-motion";
import { useAlgorithms, useProcessingStats } from "@/hooks/use-api";
import { cn } from "@/lib/utils";

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.08,
    },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 24 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring" as const,
      stiffness: 100,
      damping: 20,
    },
  },
};

export default function Dashboard() {
  const { data: algorithms, isLoading: algorithmsLoading, error: algorithmsError } = useAlgorithms();
  const { data: stats, error: statsError } = useProcessingStats();

  const apiConnected = !algorithmsError && !statsError && algorithms !== undefined;
  const algorithmsCount = algorithms ? Object.keys(algorithms).length : 0;

  const statsData = [
    {
      title: "Imagens Processadas",
      value: stats?.total_images_processed?.toString() || "0",
      change: "+12%",
      trend: "up",
      icon: FileImage,
    },
    {
      title: "Algoritmos Ativos",
      value: algorithmsCount.toString(),
      change: `${algorithmsCount} disponíveis`,
      trend: "neutral",
      icon: Cpu,
    },
    {
      title: "Jobs em Execução",
      value: stats?.total_jobs?.toString() || "0",
      change: "+8%",
      trend: "up",
      icon: Layers,
    },
    {
      title: "Status da API",
      value: apiConnected ? "Online" : "Offline",
      change: apiConnected ? "Conectado" : "Desconectado",
      trend: apiConnected ? "up" : "down",
      icon: Activity,
    },
  ];

  const recentActivities = [
    {
      type: "process",
      title: "Compressão SVD aplicada",
      description: "brain.jpg processada com k=50",
      time: "2 min",
      algorithm: "SVD",
    },
    {
      type: "compare",
      title: "Comparação de algoritmos",
      description: "SVD vs Autoencoder vs CS",
      time: "15 min",
      algorithms: ["SVD", "AE", "CS"],
    },
    {
      type: "upload",
      title: "Nova imagem carregada",
      description: "chest_xray_001.png (512×512)",
      time: "1h",
    },
  ];

  const quickActions = [
    {
      title: "Upload",
      description: "Carregar novas imagens",
      icon: Upload,
      href: "/upload",
    },
    {
      title: "Processar",
      description: "Aplicar algoritmos",
      icon: Cpu,
      href: "/process",
    },
    {
      title: "Comparar",
      description: "Análise lado a lado",
      icon: GitCompare,
      href: "/compare",
    },
    {
      title: "Resultados",
      description: "Ver métricas",
      icon: BarChart3,
      href: "/results",
    },
  ];

  return (
    <motion.div 
      className="flex-1 space-y-8 p-6 md:p-10"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-1">
          <p className="text-caption">Bem-vindo de volta</p>
          <h1 className="text-display">Dashboard</h1>
          <p className="text-muted-foreground max-w-md">
            Visão geral do seu processamento de imagens
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge 
            variant="secondary" 
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 rounded-full border",
              apiConnected 
                ? "bg-transparent border-foreground/20 text-foreground" 
                : "bg-transparent border-destructive/50 text-destructive"
            )}
          >
            <span className={cn(
              "relative flex h-2 w-2",
              apiConnected ? "text-foreground" : "text-destructive"
            )}>
              {apiConnected && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
              )}
              <span className="relative inline-flex rounded-full h-2 w-2 bg-current" />
            </span>
            {apiConnected ? "Sistema Online" : "Sistema Offline"}
          </Badge>
        </div>
      </motion.div>

      {/* Stats Grid */}
      <motion.div variants={itemVariants} className="grid gap-4 md:gap-6 grid-cols-2 lg:grid-cols-4">
        {statsData.map((stat, index) => (
          <motion.div
            key={stat.title}
            variants={itemVariants}
            whileHover={{ y: -4 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            <Card className="metric-card group">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="p-2.5 rounded-xl bg-accent border border-border group-hover:bg-foreground group-hover:border-foreground transition-all duration-300">
                    <stat.icon className="h-5 w-5 text-foreground group-hover:text-background transition-colors" strokeWidth={1.5} />
                  </div>
                  {stat.trend === "up" && (
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <TrendingUp className="h-3 w-3" />
                      {stat.change}
                    </div>
                  )}
                </div>
                <div className="space-y-1">
                  <p className="text-3xl font-semibold tracking-tight">
                    {stat.value}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {stat.title}
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Quick Actions */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <Card className="border-border bg-card/50">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <CardTitle className="text-xl font-semibold">Ações Rápidas</CardTitle>
                  <CardDescription>
                    Comece a processar suas imagens
                  </CardDescription>
                </div>
                <Zap className="h-5 w-5 text-muted-foreground" strokeWidth={1.5} />
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 sm:grid-cols-2">
                {quickActions.map((action, index) => (
                  <Link key={action.title} href={action.href}>
                    <motion.div
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className="group"
                    >
                      <div className={cn(
                        "relative overflow-hidden rounded-xl border border-border p-5 transition-all duration-300",
                        "hover:border-foreground/20 hover:bg-accent/50 bg-background"
                      )}>
                        <div className="flex items-start justify-between">
                          <div className="flex items-start gap-4">
                            <div className="p-2.5 rounded-xl bg-accent border border-border group-hover:bg-foreground group-hover:border-foreground transition-all duration-300">
                              <action.icon className="h-5 w-5 text-foreground group-hover:text-background transition-colors" strokeWidth={1.5} />
                            </div>
                            <div className="space-y-1">
                              <h3 className="font-medium text-foreground">
                                {action.title}
                              </h3>
                              <p className="text-sm text-muted-foreground">
                                {action.description}
                              </p>
                            </div>
                          </div>
                          <ArrowUpRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-all duration-300" />
                        </div>
                      </div>
                    </motion.div>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Recent Activity */}
        <motion.div variants={itemVariants}>
          <Card className="border-border bg-card/50 h-full">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <CardTitle className="text-xl font-semibold">Atividade</CardTitle>
                  <CardDescription>
                    Últimas ações
                  </CardDescription>
                </div>
                <Clock className="h-5 w-5 text-muted-foreground" strokeWidth={1.5} />
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentActivities.map((activity, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: 16 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 + 0.3 }}
                    className="flex items-start gap-4 p-3 rounded-xl hover:bg-accent/50 transition-colors group"
                  >
                    <div className="p-2 rounded-lg bg-accent border border-border">
                      {activity.type === "process" && <Cpu className="h-4 w-4" strokeWidth={1.5} />}
                      {activity.type === "compare" && <GitCompare className="h-4 w-4" strokeWidth={1.5} />}
                      {activity.type === "upload" && <Upload className="h-4 w-4" strokeWidth={1.5} />}
                    </div>
                    <div className="flex-1 min-w-0 space-y-1">
                      <p className="text-sm font-medium truncate">
                        {activity.title}
                      </p>
                      <p className="text-xs text-muted-foreground truncate">
                        {activity.description}
                      </p>
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-[10px] text-muted-foreground/60">
                          {activity.time}
                        </span>
                        {activity.algorithms && activity.algorithms.map((alg) => (
                          <Badge 
                            key={alg} 
                            variant="secondary" 
                            className="text-[9px] px-1.5 py-0 bg-accent text-foreground border border-border rounded-md"
                          >
                            {alg}
                          </Badge>
                        ))}
                        {activity.algorithm && (
                          <Badge 
                            variant="secondary" 
                            className="text-[9px] px-1.5 py-0 bg-accent text-foreground border border-border rounded-md"
                          >
                            {activity.algorithm}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Algorithms Section */}
      <motion.div variants={itemVariants}>
        <Card className="border-border bg-card/50 overflow-hidden">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <CardTitle className="text-xl font-semibold">Algoritmos</CardTitle>
                <CardDescription>
                  {algorithmsLoading
                    ? "Carregando..."
                    : algorithmsError
                      ? "Erro de conexão"
                      : `${algorithmsCount} algoritmos disponíveis`
                  }
                </CardDescription>
              </div>
              <Box className="h-5 w-5 text-muted-foreground" strokeWidth={1.5} />
            </div>
          </CardHeader>
          
          <CardContent>
            {algorithmsLoading ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="p-5 rounded-xl border border-border bg-accent/30">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 rounded-lg bg-muted animate-pulse" />
                      <div className="flex-1 space-y-2">
                        <div className="h-4 w-24 rounded bg-muted animate-pulse" />
                        <div className="h-3 w-16 rounded bg-muted animate-pulse" />
                      </div>
                    </div>
                    <div className="h-3 w-full rounded bg-muted animate-pulse" />
                  </div>
                ))}
              </div>
            ) : algorithmsError ? (
              <div className="text-center py-16">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent border border-border mb-4">
                  <XCircle className="h-8 w-8 text-muted-foreground" strokeWidth={1.5} />
                </div>
                <h3 className="text-lg font-semibold mb-2">
                  Erro de Conexão
                </h3>
                <p className="text-sm text-muted-foreground max-w-sm mx-auto">
                  Não foi possível conectar com o backend. Verifique se o servidor está rodando na porta 8001.
                </p>
              </div>
            ) : algorithms && algorithmsCount > 0 ? (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {Object.values(algorithms).slice(0, 8).map((alg: any, index) => (
                  <motion.div
                    key={alg.type}
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 + 0.2 }}
                    whileHover={{ y: -4 }}
                    className="group cursor-pointer"
                  >
                    <div className={cn(
                      "p-5 rounded-xl border border-border transition-all duration-300",
                      "hover:border-foreground/20 hover:bg-accent/50 bg-background"
                    )}>
                      <div className="flex items-center gap-3 mb-3">
                        <span className="text-2xl group-hover:scale-110 transition-transform">
                          {alg.icon || "⚙️"}
                        </span>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-sm truncate">
                            {alg.name}
                          </h4>
                          <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-mono">
                            {alg.type}
                          </p>
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground line-clamp-2">
                        {alg.description || "Algoritmo de processamento de imagens"}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="text-center py-16">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent border border-border mb-4">
                  <Cpu className="h-8 w-8 text-muted-foreground" strokeWidth={1.5} />
                </div>
                <h3 className="text-lg font-semibold mb-2">Nenhum Algoritmo</h3>
                <p className="text-sm text-muted-foreground">
                  Os algoritmos serão carregados automaticamente
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}
