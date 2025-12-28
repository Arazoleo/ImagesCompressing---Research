"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ImageUpload } from "@/components/image/image-upload";
import { ImageFile } from "@/lib/types";
import { ArrowRight, Info, Sparkles, FileImage, HardDrive, Layers, CheckCircle2, ArrowUpRight } from "lucide-react";
import Link from "next/link";
import { motion, Variants } from "framer-motion";
import { cn } from "@/lib/utils";

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: "spring", stiffness: 100, damping: 15 },
  },
};

export default function UploadPage() {
  const [uploadedImages, setUploadedImages] = useState<ImageFile[]>([]);

  const handleImagesUploaded = (images: ImageFile[]) => {
    setUploadedImages(images);
  };

  const algorithmCards = [
    {
      name: "SVD Compression",
      description: "Decomposição matricial clássica para compressão eficiente",
      icon: "🧮",
      gradient: "from-blue-500 to-cyan-500",
    },
    {
      name: "Autoencoder",
      description: "Redes neurais para aprendizado de representações",
      icon: "🧠",
      gradient: "from-violet-500 to-purple-500",
    },
    {
      name: "Compressed Sensing",
      description: "Amostragem esparsa com reconstrução L1",
      icon: "📡",
      gradient: "from-amber-500 to-orange-500",
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
              Upload de Imagens
            </span>
          </h1>
          <p className="text-muted-foreground mt-1">
            Carregue suas imagens para começar o processamento
          </p>
        </div>
      </motion.div>

      {/* Info Card */}
      <motion.div variants={itemVariants}>
        <Card className="relative overflow-hidden border-border/50 bg-gradient-to-br from-blue-500/5 via-cyan-500/5 to-transparent">
          <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-blue-500/10 to-cyan-500/5 rounded-full -translate-y-32 translate-x-32 blur-3xl" />
          <CardHeader className="relative">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 shadow-lg">
                <Info className="h-5 w-5 text-white" />
              </div>
              <CardTitle>Dicas para Upload</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="relative">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-3">
                <h4 className="font-semibold text-sm flex items-center gap-2">
                  <FileImage className="h-4 w-4 text-blue-500" />
                  Formatos Suportados
                </h4>
                <ul className="text-sm text-muted-foreground space-y-2">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                    JPEG, PNG, GIF, WebP, BMP
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                    Máximo de 10 imagens por vez
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                    Tamanho máximo: 10MB por arquivo
                  </li>
                </ul>
              </div>
              <div className="space-y-3">
                <h4 className="font-semibold text-sm flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-amber-500" />
                  Recomendações
                </h4>
                <ul className="text-sm text-muted-foreground space-y-2">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                    Imagens de alta qualidade para melhores resultados
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                    Formatos sem compressão quando possível
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                    Resoluções entre 512x512 e 2048x2048
                  </li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Upload Component */}
      <motion.div variants={itemVariants}>
        <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary/80 shadow-lg">
                <ArrowUpRight className="h-5 w-5 text-primary-foreground" />
              </div>
              <div>
                <CardTitle>Selecionar Imagens</CardTitle>
                <CardDescription>
                  Arraste e solte suas imagens aqui ou clique para selecionar
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <ImageUpload
              onImagesUploaded={handleImagesUploaded}
              maxFiles={10}
              maxSize={10 * 1024 * 1024}
            />
          </CardContent>
        </Card>
      </motion.div>

      {/* Next Steps */}
      {uploadedImages.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 100, damping: 15 }}
        >
          <Card className="relative overflow-hidden border-border/50 bg-gradient-to-br from-emerald-500/5 via-green-500/5 to-transparent">
            <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-emerald-500/10 to-green-500/5 rounded-full -translate-y-32 translate-x-32 blur-3xl" />
            <CardHeader className="relative">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-green-500 shadow-lg">
                  <CheckCircle2 className="h-5 w-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-emerald-700 dark:text-emerald-400">Próximos Passos</CardTitle>
                  <CardDescription>
                    {uploadedImages.length} imagem(ns) carregada(s) com sucesso!
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative space-y-6">
              <div className="flex flex-col sm:flex-row gap-4">
                <Link href="/process">
                  <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <Button className="w-full sm:w-auto bg-gradient-to-r from-emerald-500 to-green-500 hover:from-emerald-600 hover:to-green-600 text-white border-0 shadow-lg shadow-emerald-500/25">
                      <span>Processar Imagens</span>
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </Button>
                  </motion.div>
                </Link>
                <Link href="/compare">
                  <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <Button variant="outline" className="w-full sm:w-auto">
                      Comparar Algoritmos
                    </Button>
                  </motion.div>
                </Link>
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                {algorithmCards.map((alg, index) => (
                  <motion.div
                    key={alg.name}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.03, y: -2 }}
                    className="group cursor-pointer"
                  >
                    <Card className="border-border/50 bg-background/80 backdrop-blur-sm hover:shadow-lg transition-all duration-300">
                      <CardContent className="p-4 text-center">
                        <span className="text-3xl mb-2 block group-hover:scale-110 transition-transform">
                          {alg.icon}
                        </span>
                        <h4 className="font-semibold text-sm group-hover:text-primary transition-colors">
                          {alg.name}
                        </h4>
                        <p className="text-xs text-muted-foreground mt-1">
                          {alg.description}
                        </p>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Statistics */}
      <motion.div variants={itemVariants} className="grid gap-4 sm:grid-cols-3">
        {[
          {
            title: "Imagens Hoje",
            value: uploadedImages.length,
            subtitle: "carregadas nesta sessão",
            icon: FileImage,
            gradient: "from-blue-500 to-cyan-500",
          },
          {
            title: "Espaço Total",
            value: `${(uploadedImages.reduce((acc, img) => acc + img.size, 0) / 1024 / 1024).toFixed(1)} MB`,
            subtitle: "de dados carregados",
            icon: HardDrive,
            gradient: "from-violet-500 to-purple-500",
          },
          {
            title: "Formatos",
            value: new Set(uploadedImages.map(img => img.file.type)).size,
            subtitle: "tipos de arquivo diferentes",
            icon: Layers,
            gradient: "from-amber-500 to-orange-500",
          },
        ].map((stat, index) => (
          <motion.div
            key={stat.title}
            whileHover={{ scale: 1.02, y: -2 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-muted-foreground">
                      {stat.title}
                    </p>
                    <p className="text-3xl font-bold">{stat.value}</p>
                    <p className="text-xs text-muted-foreground">
                      {stat.subtitle}
                    </p>
                  </div>
                  <div className={cn(
                    "flex items-center justify-center w-10 h-10 rounded-xl",
                    "bg-gradient-to-br shadow-lg",
                    stat.gradient
                  )}>
                    <stat.icon className="h-5 w-5 text-white" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>
    </motion.div>
  );
}
