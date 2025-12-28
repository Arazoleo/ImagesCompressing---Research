"use client";

import { useState } from "react";
import { ImageGallery } from "@/components/gallery/image-gallery";
import { ImageViewer } from "@/components/image/image-viewer";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Card, CardContent } from "@/components/ui/card";
import { ImageFile } from "@/lib/types";
import { ArrowLeft, Download, Cpu, FileImage, Sparkles } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";

export default function GalleryPage() {
  const [selectedImage, setSelectedImage] = useState<ImageFile | null>(null);
  const [viewerOpen, setViewerOpen] = useState(false);

  const handleImageSelect = (image: ImageFile) => {
    setSelectedImage(image);
    setViewerOpen(true);
  };

  return (
    <motion.div 
      className="flex-1 space-y-8 p-6 md:p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <Link href="/">
            <Button variant="outline" size="sm" className="rounded-xl">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Voltar
            </Button>
          </Link>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
              <span className="bg-gradient-to-r from-foreground via-foreground/90 to-foreground/70 bg-clip-text text-transparent">
                Galeria de Imagens
              </span>
            </h1>
            <p className="text-muted-foreground mt-1">
              Explore todas as suas imagens carregadas e processadas
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-muted/30 rounded-xl px-4 py-2">
            <FileImage className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">0 imagens</span>
          </div>
        </div>
      </div>

      {/* Gallery Info */}
      <Card className="relative overflow-hidden border-border/50 bg-gradient-to-br from-primary/5 via-transparent to-primary/5">
        <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-primary/10 to-transparent rounded-full -translate-y-32 translate-x-32 blur-3xl" />
        <CardContent className="relative p-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center shadow-lg">
              <Sparkles className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h3 className="font-semibold text-lg">Sua Biblioteca de Imagens</h3>
              <p className="text-sm text-muted-foreground">
                Visualize, organize e processe suas imagens com algoritmos avançados
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Gallery */}
      <Card className="border-border/50 shadow-lg bg-card/50 backdrop-blur-sm">
        <CardContent className="p-6">
          <ImageGallery
            onImageSelect={handleImageSelect}
            showActions={true}
          />
        </CardContent>
      </Card>

      {/* Image Viewer Dialog */}
      <Dialog open={viewerOpen} onOpenChange={setViewerOpen}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-hidden bg-background/95 backdrop-blur-xl border-border/50">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center">
                <FileImage className="h-4 w-4 text-primary-foreground" />
              </div>
              {selectedImage?.name}
            </DialogTitle>
          </DialogHeader>

          {selectedImage && (
            <div className="space-y-6">
              <div className="rounded-xl overflow-hidden bg-muted/30">
                <ImageViewer
                  originalImage={selectedImage}
                  className="w-full"
                />
              </div>

              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 pt-4 border-t border-border/50">
                <div className="text-sm text-muted-foreground space-y-1">
                  <p className="flex items-center gap-2">
                    <span className="font-medium text-foreground">Dimensões:</span> 
                    {selectedImage.width}×{selectedImage.height} px
                  </p>
                  <p className="flex items-center gap-2">
                    <span className="font-medium text-foreground">Tamanho:</span> 
                    {(selectedImage.size / 1024).toFixed(1)} KB
                  </p>
                </div>

                <div className="flex gap-3">
                  <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <Button variant="outline" className="rounded-xl">
                      <Download className="h-4 w-4 mr-2" />
                      Download Original
                    </Button>
                  </motion.div>
                  <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <Link href="/process">
                      <Button className="rounded-xl bg-gradient-to-r from-primary to-primary/80">
                        <Cpu className="h-4 w-4 mr-2" />
                        Processar Imagem
                      </Button>
                    </Link>
                  </motion.div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}
