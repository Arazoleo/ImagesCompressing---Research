"use client";

import { useCallback, useState } from "react";
import { useDropzone, FileRejection } from "react-dropzone";
import { Upload, X, FileImage, AlertCircle, CheckCircle2, ImagePlus, Sparkles } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { ImageFile } from "@/lib/types";
import { getImageDimensions, formatFileSize } from "@/lib/api";
import { useUploadImage } from "@/hooks/use-api";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

interface ImageUploadProps {
  onImagesUploaded: (images: ImageFile[]) => void;
  maxFiles?: number;
  maxSize?: number;
  acceptedTypes?: string[];
}

export function ImageUpload({
  onImagesUploaded,
  maxFiles = 10,
  maxSize = 10 * 1024 * 1024,
  acceptedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp']
}: ImageUploadProps) {
  const [uploadedImages, setUploadedImages] = useState<ImageFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const uploadImageMutation = useUploadImage();

  const processFile = async (file: File): Promise<ImageFile> => {
    try {
      // Primeiro, obter dimensões da imagem
      const { width, height } = await getImageDimensions(file);

      // Fazer upload real para o backend
      const uploadedImage = await uploadImageMutation.mutateAsync(file);

      return {
        id: uploadedImage.id,
        file,
        name: file.name,
        size: file.size,
        url: uploadedImage.url,
        width,
        height,
        uploaded_at: uploadedImage.uploaded_at
      };
    } catch (error) {
      throw new Error(`Erro ao fazer upload de ${file.name}: ${error}`);
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
    setError(null);

    if (rejectedFiles.length > 0) {
      const errors = rejectedFiles.map(({ file, errors }) =>
        `${file.name}: ${errors.map(e => e.message).join(', ')}`
      );
      setError(`Arquivos rejeitados: ${errors.join('; ')}`);
      return;
    }

    const totalFiles = uploadedImages.length + acceptedFiles.length;
    if (totalFiles > maxFiles) {
      setError(`Máximo de ${maxFiles} arquivos permitido`);
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      const processedImages: ImageFile[] = [];

      for (let i = 0; i < acceptedFiles.length; i++) {
        const imageFile = await processFile(acceptedFiles[i]);
        processedImages.push(imageFile);
        setUploadProgress(((i + 1) / acceptedFiles.length) * 100);
      }

      const newImages = [...uploadedImages, ...processedImages];
      setUploadedImages(newImages);
      onImagesUploaded(newImages);

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Erro desconhecido');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }, [uploadedImages, maxFiles, onImagesUploaded]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedTypes.reduce((acc, type) => ({ ...acc, [type]: [] }), {}),
    maxSize,
    maxFiles: maxFiles - uploadedImages.length,
    disabled: uploading,
  });

  const removeImage = (id: string) => {
    const newImages = uploadedImages.filter(img => img.id !== id);
    setUploadedImages(newImages);
    onImagesUploaded(newImages);
  };

  return (
    <div className="space-y-6">
      {/* Dropzone */}
      <motion.div
        whileHover={{ scale: uploading ? 1 : 1.01 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
      >
        <div
          {...getRootProps()}
          className={cn(
            "relative cursor-pointer transition-all duration-300 rounded-2xl p-8 md:p-12 text-center overflow-hidden",
            "border-2 border-dashed",
            isDragActive
              ? "border-primary bg-primary/5 scale-[1.02]"
              : "border-border/50 hover:border-primary/50 hover:bg-muted/30",
            uploading && "pointer-events-none opacity-70"
          )}
        >
          {/* Background gradient effect */}
          <div className={cn(
            "absolute inset-0 transition-opacity duration-300",
            isDragActive ? "opacity-100" : "opacity-0"
          )}>
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-primary/5" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-primary/20 rounded-full blur-3xl" />
          </div>
          
          <input {...getInputProps()} />
          
          <div className="relative flex flex-col items-center space-y-4">
            <motion.div
              animate={isDragActive ? { scale: 1.1, rotate: 5 } : { scale: 1, rotate: 0 }}
              transition={{ type: "spring", stiffness: 300, damping: 15 }}
              className={cn(
                "relative flex items-center justify-center w-20 h-20 rounded-2xl transition-all duration-300",
                isDragActive
                  ? "bg-gradient-to-br from-primary to-primary/80 shadow-lg shadow-primary/30"
                  : "bg-gradient-to-br from-muted to-muted/50"
              )}
            >
              {isDragActive ? (
                <Sparkles className="h-8 w-8 text-primary-foreground" />
              ) : (
                <ImagePlus className="h-8 w-8 text-muted-foreground" />
              )}
              
              {/* Animated ring */}
              {isDragActive && (
                <motion.div
                  className="absolute inset-0 rounded-2xl border-2 border-primary"
                  initial={{ scale: 1, opacity: 1 }}
                  animate={{ scale: 1.5, opacity: 0 }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              )}
            </motion.div>

            <div className="space-y-2">
              <h3 className="text-lg font-semibold">
                {isDragActive ? (
                  <span className="text-primary">Solte as imagens aqui</span>
                ) : (
                  "Arraste e solte imagens"
                )}
              </h3>
              <p className="text-sm text-muted-foreground">
                ou clique para selecionar arquivos
              </p>
            </div>
            
            <div className="flex flex-wrap justify-center gap-2">
              {['JPEG', 'PNG', 'GIF', 'WebP', 'BMP'].map((format) => (
                <Badge
                  key={format}
                  variant="secondary"
                  className="bg-muted/50 text-muted-foreground text-xs"
                >
                  {format}
                </Badge>
              ))}
            </div>
            
            <p className="text-xs text-muted-foreground">
              Máx. {maxFiles} arquivos • Máx. {formatFileSize(maxSize)} cada
            </p>

            {/* Upload Progress */}
            <AnimatePresence>
              {uploading && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="w-full max-w-xs space-y-2"
                >
                  <div className="relative">
                    <Progress value={uploadProgress} className="h-2" />
                    <div 
                      className="absolute top-0 left-0 h-2 rounded-full bg-gradient-to-r from-primary via-primary/80 to-primary animate-shimmer"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <p className="text-sm text-muted-foreground flex items-center justify-center gap-2">
                    <motion.span
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="inline-block w-4 h-4 border-2 border-primary border-t-transparent rounded-full"
                    />
                    Processando... {Math.round(uploadProgress)}%
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>

      {/* Error Alert */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
          >
            <Alert variant="destructive" className="border-red-500/50 bg-red-500/10">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Uploaded Images Preview */}
      <AnimatePresence>
        {uploadedImages.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="space-y-4"
          >
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                Imagens Carregadas
              </h3>
              <Badge variant="secondary" className="bg-primary/10 text-primary border-0">
                {uploadedImages.length}/{maxFiles}
              </Badge>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <AnimatePresence mode="popLayout">
                {uploadedImages.map((image, index) => (
                  <motion.div
                    key={image.id}
                    layout
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{ type: "spring", stiffness: 300, damping: 25, delay: index * 0.05 }}
                  >
                    <Card className="group overflow-hidden border-border/50 hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5 transition-all duration-300">
                      <div className="relative aspect-video overflow-hidden bg-muted">
                        <img
                          src={image.url}
                          alt={image.name}
                          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                        <motion.div
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                        >
                          <Button
                            variant="destructive"
                            size="icon"
                            className="absolute top-2 right-2 h-8 w-8 opacity-0 group-hover:opacity-100 transition-all duration-300 rounded-lg shadow-lg"
                            onClick={(e) => {
                              e.stopPropagation();
                              removeImage(image.id);
                            }}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </motion.div>
                        
                        {/* Image info overlay */}
                        <div className="absolute bottom-0 left-0 right-0 p-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                          <p className="text-white text-xs font-medium truncate">
                            {image.width}×{image.height} px
                          </p>
                        </div>
                      </div>

                      <CardContent className="p-4">
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <FileImage className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                            <p className="text-sm font-medium truncate" title={image.name}>
                              {image.name}
                            </p>
                          </div>

                          <div className="flex items-center justify-between text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              {image.width}×{image.height}
                            </span>
                            <Badge variant="secondary" className="text-xs bg-muted/50">
                              {formatFileSize(image.size)}
                            </Badge>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
