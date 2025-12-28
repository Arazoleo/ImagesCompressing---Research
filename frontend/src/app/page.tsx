"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import {
  Upload,
  Cpu,
  BarChart3,
  GitCompare,
  FileImage,
  TrendingUp,
  Clock,
  Zap,
  ArrowRight,
  CheckCircle,
  XCircle,
  Sparkles,
  Activity,
  Layers,
  ArrowUpRight,
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
      staggerChildren: 0.1,
    },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring" as const,
      stiffness: 100,
      damping: 15,
    },
  },
};

export default function Dashboard() {
  const { data: algorithms, isLoading: algorithmsLoading, error: algorithmsError } = useAlgorithms();
  const { data: stats, isLoading: statsLoading, error: statsError } = useProcessingStats();

  const apiConnected = !algorithmsError && !statsError && algorithms !== undefined;
  const algorithmsCount = algorithms ? Object.keys(algorithms).length : 0;

  const statsData = [
    {
      title: "Imagens Processadas",
      value: stats?.total_images_processed?.toString() || "0",
      change: "+12%",
      trend: "up",
      icon: FileImage,
      gradient: "from-blue-500 to-cyan-500",
      bgGradient: "from-blue-500/10 to-cyan-500/5",
    },
    {
      title: "Algoritmos Disponíveis",
      value: algorithmsCount.toString(),
      change: `${algorithmsCount} ativos`,
      trend: "neutral",
      icon: Cpu,
      gradient: "from-violet-500 to-purple-500",
      bgGradient: "from-violet-500/10 to-purple-500/5",
    },
    {
      title: "Jobs de Processamento",
      value: stats?.total_jobs?.toString() || "0",
      change: "+8%",
      trend: "up",
      icon: Layers,
      gradient: "from-amber-500 to-orange-500",
      bgGradient: "from-amber-500/10 to-orange-500/5",
    },
    {
      title: "Status da API",
      value: apiConnected ? "Online" : "Offline",
      change: apiConnected ? "Conectado" : "Desconectado",
      trend: apiConnected ? "up" : "down",
      icon: Activity,
      gradient: apiConnected ? "from-emerald-500 to-green-500" : "from-red-500 to-rose-500",
      bgGradient: apiConnected ? "from-emerald-500/10 to-green-500/5" : "from-red-500/10 to-rose-500/5",
    },
  ];

  const recentActivities = [
    {
      type: "process",
      title: "Compressão SVD aplicada",
      description: "brain.jpg processada com k=50",
      time: "2 minutos atrás",
      algorithm: "SVD",
      status: "success",
    },
    {
      type: "compare",
      title: "Comparação de algoritmos",
      description: "SVD vs Autoencoder vs Compressed Sensing",
      time: "15 minutos atrás",
      algorithms: ["SVD", "AE", "CS"],
      status: "success",
    },
    {
      type: "upload",
      title: "Nova imagem carregada",
      description: "chest_xray_001.png (512x512)",
      time: "1 hora atrás",
      status: "success",
    },
  ];

  const quickActions = [
    {
      title: "Upload de Imagem",
      description: "Carregue novas imagens para processamento",
      icon: Upload,
      href: "/upload",
      gradient: "from-blue-500 to-cyan-500",
      hoverGradient: "group-hover:from-blue-600 group-hover:to-cyan-600",
    },
    {
      title: "Processar Imagem",
      description: "Aplique algoritmos de compressão",
      icon: Cpu,
      href: "/process",
      gradient: "from-violet-500 to-purple-500",
      hoverGradient: "group-hover:from-violet-600 group-hover:to-purple-600",
    },
    {
      title: "Comparar Algoritmos",
      description: "Compare performance lado a lado",
      icon: GitCompare,
      href: "/compare",
      gradient: "from-amber-500 to-orange-500",
      hoverGradient: "group-hover:from-amber-600 group-hover:to-orange-600",
    },
    {
      title: "Ver Resultados",
      description: "Análise detalhada de métricas",
      icon: BarChart3,
      href: "/results",
      gradient: "from-emerald-500 to-green-500",
      hoverGradient: "group-hover:from-emerald-600 group-hover:to-green-600",
    },
  ];

  return (
    <motion.div 
      className="flex-1 space-y-8 p-6 md:p-8"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
            <span className="bg-gradient-to-r from-foreground via-foreground/90 to-foreground/70 bg-clip-text text-transparent">
              Dashboard
            </span>
          </h1>
          <p className="text-muted-foreground mt-1">
            Visão geral do seu processamento de imagens
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge 
            variant="secondary" 
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-full",
              "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-0"
            )}
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            Sistema Online
          </Badge>
          <Badge
            variant={algorithmsError || statsError ? "destructive" : "secondary"}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-full border-0",
              algorithmsError || statsError 
                ? "bg-red-500/10 text-red-600 dark:text-red-400" 
                : "bg-primary/10 text-primary"
            )}
          >
            {algorithmsError || statsError ? (
              <XCircle className="h-3.5 w-3.5" />
            ) : (
              <CheckCircle className="h-3.5 w-3.5" />
            )}
            API {algorithmsError || statsError ? "Offline" : "Conectada"}
          </Badge>
        </div>
      </motion.div>

      {/* Stats Grid */}
      <motion.div variants={itemVariants} className="grid gap-4 md:gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        {statsData.map((stat, index) => (
          <motion.div
            key={stat.title}
            variants={itemVariants}
            whileHover={{ scale: 1.02, y: -4 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            <Card className={cn(
              "relative overflow-hidden border-0 shadow-lg",
              "bg-gradient-to-br",
              stat.bgGradient,
              "backdrop-blur-sm"
            )}>
              <div className="absolute inset-0 bg-gradient-to-br from-background/50 to-background/80" />
              <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br opacity-20 rounded-full -translate-y-16 translate-x-16 blur-2xl"
                style={{ background: `linear-gradient(to bottom right, var(--${stat.gradient.split(' ')[1].replace('to-', '')}))` }} 
              />
              <CardContent className="relative p-6">
                <div className="flex items-start justify-between">
                  <div className="space-y-3">
                    <p className="text-sm font-medium text-muted-foreground">
                      {stat.title}
                    </p>
                    <p className="text-3xl font-bold tracking-tight">
                      {stat.value}
                    </p>
                    <div className="flex items-center gap-1.5 text-xs">
                      {stat.trend === "up" && (
                        <TrendingUp className="h-3.5 w-3.5 text-emerald-500" />
                      )}
                      <span className={cn(
                        stat.trend === "up" && "text-emerald-600 dark:text-emerald-400",
                        stat.trend === "down" && "text-red-600 dark:text-red-400",
                        stat.trend === "neutral" && "text-muted-foreground"
                      )}>
                        {stat.change}
                      </span>
                    </div>
                  </div>
                  <div className={cn(
                    "flex items-center justify-center w-12 h-12 rounded-xl",
                    "bg-gradient-to-br shadow-lg",
                    stat.gradient
                  )}>
                    <stat.icon className="h-6 w-6 text-white" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-7">
        {/* Quick Actions */}
        <motion.div variants={itemVariants} className="lg:col-span-4">
          <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
            <CardHeader className="pb-4">
              <div className="flex items-center gap-2">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
                  <Zap className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <CardTitle>Ações Rápidas</CardTitle>
                  <CardDescription>
                    Comece a processar suas imagens
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-2">
                {quickActions.map((action, index) => (
                  <Link key={action.title} href={action.href}>
                    <motion.div
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className="group"
                    >
                      <Card className={cn(
                        "relative overflow-hidden border-border/50 transition-all duration-300",
                        "hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5",
                        "bg-gradient-to-br from-muted/30 to-muted/10"
                      )}>
                        <CardContent className="p-4">
                          <div className="flex items-start gap-4">
                            <div className={cn(
                              "flex items-center justify-center w-12 h-12 rounded-xl",
                              "bg-gradient-to-br shadow-lg transition-all duration-300",
                              action.gradient,
                              action.hoverGradient
                            )}>
                              <action.icon className="h-5 w-5 text-white" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
                                {action.title}
                              </h3>
                              <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                                {action.description}
                              </p>
                            </div>
                            <ArrowUpRight className="h-4 w-4 text-muted-foreground opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Recent Activity */}
        <motion.div variants={itemVariants} className="lg:col-span-3">
          <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm h-full">
            <CardHeader className="pb-4">
              <div className="flex items-center gap-2">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-amber-500/10">
                  <Clock className="h-4 w-4 text-amber-500" />
                </div>
                <div>
                  <CardTitle>Atividade Recente</CardTitle>
                  <CardDescription>
                    Últimas ações realizadas
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentActivities.map((activity, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 + 0.3 }}
                    className="flex items-start gap-4 p-3 rounded-xl hover:bg-muted/50 transition-colors group"
                  >
                    <div className={cn(
                      "flex items-center justify-center w-10 h-10 rounded-xl shadow-md",
                      activity.type === "process" && "bg-gradient-to-br from-emerald-500 to-green-500",
                      activity.type === "compare" && "bg-gradient-to-br from-violet-500 to-purple-500",
                      activity.type === "upload" && "bg-gradient-to-br from-blue-500 to-cyan-500"
                    )}>
                      {activity.type === "process" && <Cpu className="h-4 w-4 text-white" />}
                      {activity.type === "compare" && <GitCompare className="h-4 w-4 text-white" />}
                      {activity.type === "upload" && <Upload className="h-4 w-4 text-white" />}
                    </div>
                    <div className="flex-1 min-w-0 space-y-1">
                      <p className="text-sm font-medium truncate group-hover:text-primary transition-colors">
                        {activity.title}
                      </p>
                      <p className="text-xs text-muted-foreground truncate">
                        {activity.description}
                      </p>
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-[10px] text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                          {activity.time}
                        </span>
                        {activity.algorithms && activity.algorithms.map((alg) => (
                          <Badge 
                            key={alg} 
                            variant="secondary" 
                            className="text-[10px] px-1.5 py-0 bg-primary/10 text-primary border-0"
                          >
                            {alg}
                          </Badge>
                        ))}
                        {activity.algorithm && (
                          <Badge 
                            variant="secondary" 
                            className="text-[10px] px-1.5 py-0 bg-primary/10 text-primary border-0"
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
        <Card className="relative overflow-hidden border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/5" />
          <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-primary/10 to-transparent rounded-full -translate-y-48 translate-x-48 blur-3xl" />
          
          <CardHeader className="relative">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary/80 shadow-lg">
                <Sparkles className="h-5 w-5 text-primary-foreground" />
              </div>
              <div>
                <CardTitle className="text-xl">Algoritmos Disponíveis</CardTitle>
                <CardDescription>
                  {algorithmsLoading
                    ? "Carregando algoritmos..."
                    : algorithmsError
                      ? "Erro ao conectar com a API"
                      : `${algorithmsCount} algoritmos prontos para uso`
                  }
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          
          <CardContent className="relative">
            {algorithmsLoading ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="p-4 rounded-xl bg-muted/30 animate-pulse">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 rounded-lg bg-muted" />
                      <div className="flex-1 space-y-2">
                        <div className="h-4 w-24 rounded bg-muted" />
                        <div className="h-3 w-16 rounded bg-muted" />
                      </div>
                    </div>
                    <div className="h-3 w-full rounded bg-muted" />
                  </div>
                ))}
              </div>
            ) : algorithmsError ? (
              <div className="text-center py-12">
                <div className="relative inline-block">
                  <div className="absolute inset-0 bg-red-500/20 rounded-full blur-xl" />
                  <div className="relative flex items-center justify-center w-20 h-20 rounded-full bg-red-500/10 mx-auto mb-4">
                    <XCircle className="h-10 w-10 text-red-500" />
                  </div>
                </div>
                <h3 className="text-lg font-semibold mb-2 text-red-600 dark:text-red-400">
                  Erro de Conexão
                </h3>
                <p className="text-muted-foreground mb-4">
                  Não foi possível conectar com o backend
                </p>
                <p className="text-sm text-muted-foreground">
                  Verifique se o servidor FastAPI está rodando na porta 8001
                </p>
              </div>
            ) : algorithms && algorithmsCount > 0 ? (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {Object.values(algorithms).slice(0, 8).map((alg: any, index) => (
                  <motion.div
                    key={alg.type}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 + 0.3 }}
                    whileHover={{ scale: 1.03, y: -2 }}
                    className="group cursor-pointer"
                  >
                    <div className={cn(
                      "p-4 rounded-xl border border-border/50 transition-all duration-300",
                      "bg-gradient-to-br from-muted/30 to-muted/10",
                      "hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5"
                    )}>
                      <div className="flex items-center gap-3 mb-3">
                        <span className="text-2xl group-hover:scale-110 transition-transform">
                          {alg.icon || "⚙️"}
                        </span>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-sm truncate group-hover:text-primary transition-colors">
                            {alg.name}
                          </h4>
                          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">
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
              <div className="text-center py-12">
                <div className="relative inline-block">
                  <div className="absolute inset-0 bg-muted/50 rounded-full blur-xl" />
                  <div className="relative flex items-center justify-center w-20 h-20 rounded-full bg-muted mx-auto mb-4">
                    <Cpu className="h-10 w-10 text-muted-foreground" />
                  </div>
                </div>
                <h3 className="text-lg font-semibold mb-2">Nenhum Algoritmo</h3>
                <p className="text-muted-foreground">
                  Os algoritmos serão carregados automaticamente quando disponíveis
                </p>
              </div>
            )}
            
            {algorithmsCount > 8 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className="mt-6 text-center"
              >
                <Link href="/algorithms">
                  <Button variant="outline" className="rounded-full">
                    Ver todos os {algorithmsCount} algoritmos
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
              </motion.div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}
