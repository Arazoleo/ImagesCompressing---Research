"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ImageUpload } from "@/components/image/image-upload";
import { ImageFile } from "@/lib/types";
import { ArrowRight, Info, FileImage, HardDrive, Layers, CheckCircle2, ArrowUpRight, Sparkles } from "lucide-react";
import Link from "next/link";
import { motion, Variants } from "framer-motion";
import { cn } from "@/lib/utils";

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 24 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: "spring", stiffness: 100, damping: 20 },
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
      description: "Decomposição matricial clássica",
      icon: "🧮",
    },
    {
      name: "Autoencoder",
      description: "Redes neurais profundas",
      icon: "🧠",
    },
    {
      name: "Compressed Sensing",
      description: "Amostragem esparsa L1",
      icon: "📡",
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
      <motion.div variants={itemVariants} className="space-y-1">
        <p className="text-caption">Início</p>
        <h1 className="text-headline">Upload de Imagens</h1>
        <p className="text-muted-foreground">
          Carregue suas imagens para começar o processamento
        </p>
      </motion.div>

      {/* Info Card */}
      <motion.div variants={itemVariants}>
        <Card className="border-border bg-card/50">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-accent border border-border">
                <Info className="h-5 w-5" strokeWidth={1.5} />
              </div>
              <CardTitle>Dicas para Upload</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-4">
                <h4 className="font-semibold text-sm flex items-center gap-2">
                  <FileImage className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
                  Formatos Suportados
                </h4>
                <ul className="text-sm text-muted-foreground space-y-2.5">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-foreground" strokeWidth={1.5} />
                    JPEG, PNG, GIF, WebP, BMP
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-foreground" strokeWidth={1.5} />
                    Máximo de 10 imagens por vez
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-foreground" strokeWidth={1.5} />
                    Tamanho máximo: 10MB por arquivo
                  </li>
                </ul>
              </div>
              <div className="space-y-4">
                <h4 className="font-semibold text-sm flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
                  Recomendações
                </h4>
                <ul className="text-sm text-muted-foreground space-y-2.5">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-foreground" strokeWidth={1.5} />
                    Imagens de alta qualidade
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-foreground" strokeWidth={1.5} />
                    Formatos sem compressão quando possível
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-foreground" strokeWidth={1.5} />
                    Resoluções entre 512×512 e 2048×2048
                  </li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Upload Component */}
      <motion.div variants={itemVariants}>
        <Card className="border-border bg-card/50">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-foreground">
                <ArrowUpRight className="h-5 w-5 text-background" strokeWidth={1.5} />
              </div>
              <div>
                <CardTitle>Selecionar Imagens</CardTitle>
                <CardDescription>
                  Arraste e solte suas imagens ou clique para selecionar
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
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 100, damping: 20 }}
        >
          <Card className="border-border bg-card/50">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-foreground">
                  <CheckCircle2 className="h-5 w-5 text-background" strokeWidth={1.5} />
                </div>
                <div>
                  <CardTitle>Próximos Passos</CardTitle>
                  <CardDescription>
                    {uploadedImages.length} imagem(ns) carregada(s) com sucesso
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex flex-col sm:flex-row gap-3">
                <Link href="/process">
                  <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <Button className="w-full sm:w-auto bg-foreground text-background hover:bg-foreground/90 rounded-xl">
                      <span>Processar Imagens</span>
                      <ArrowRight className="h-4 w-4 ml-2" strokeWidth={1.5} />
                    </Button>
                  </motion.div>
                </Link>
                <Link href="/compare">
                  <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <Button variant="outline" className="w-full sm:w-auto rounded-xl border-border hover:bg-accent">
                      Comparar Algoritmos
                    </Button>
                  </motion.div>
                </Link>
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                {algorithmCards.map((alg, index) => (
                  <motion.div
                    key={alg.name}
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ y: -4 }}
                    className="group cursor-pointer"
                  >
                    <div className="p-5 rounded-xl border border-border bg-background hover:border-foreground/20 transition-all duration-300">
                      <span className="text-2xl mb-3 block group-hover:scale-110 transition-transform">
                        {alg.icon}
                      </span>
                      <h4 className="font-medium text-sm">
                        {alg.name}
                      </h4>
                      <p className="text-xs text-muted-foreground mt-1">
                        {alg.description}
                      </p>
                    </div>
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
          },
          {
            title: "Espaço Total",
            value: `${(uploadedImages.reduce((acc, img) => acc + img.size, 0) / 1024 / 1024).toFixed(1)} MB`,
            subtitle: "de dados carregados",
            icon: HardDrive,
          },
          {
            title: "Formatos",
            value: new Set(uploadedImages.map(img => img.file.type)).size,
            subtitle: "tipos de arquivo diferentes",
            icon: Layers,
          },
        ].map((stat, index) => (
          <motion.div
            key={stat.title}
            whileHover={{ y: -4 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            <Card className="metric-card group">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="p-2.5 rounded-xl bg-accent border border-border group-hover:bg-foreground group-hover:border-foreground transition-all duration-300">
                    <stat.icon className="h-5 w-5 text-foreground group-hover:text-background transition-colors" strokeWidth={1.5} />
                  </div>
                </div>
                <div className="space-y-1">
                  <p className="text-3xl font-semibold">{stat.value}</p>
                  <p className="text-sm text-muted-foreground">{stat.title}</p>
                  <p className="text-xs text-muted-foreground/60">{stat.subtitle}</p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>
    </motion.div>
  );
}
