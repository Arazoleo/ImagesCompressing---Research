"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useImages } from "@/hooks/use-api";
import { ImageFile } from "@/lib/types";
import { Eye, Download, Trash2, Grid, List, Filter } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface ImageGalleryProps {
  onImageSelect?: (image: ImageFile) => void;
  selectedImageId?: string;
  showActions?: boolean;
}

export function ImageGallery({
  onImageSelect,
  selectedImageId,
  showActions = true
}: ImageGalleryProps) {
  const { data: images, isLoading, error } = useImages(50, 0);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filter, setFilter] = useState<string>('all');

  const filteredImages = images?.filter(img => {
    if (filter === 'all') return true;
    // Aqui podemos adicionar filtros por tipo, tamanho, etc.
    return true;
  }) || [];

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-48" />
          <div className="flex gap-2">
            <Skeleton className="h-8 w-8" />
            <Skeleton className="h-8 w-8" />
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {Array.from({ length: 8 }).map((_, i) => (
            <Card key={i} className="overflow-hidden">
              <Skeleton className="h-48 w-full" />
              <CardContent className="p-4">
                <Skeleton className="h-4 w-3/4 mb-2" />
                <Skeleton className="h-3 w-1/2" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="p-8 text-center">
        <div className="text-red-500 mb-4">
          <Trash2 className="h-12 w-12 mx-auto" />
        </div>
        <h3 className="text-lg font-medium mb-2">Erro ao carregar imagens</h3>
        <p className="text-muted-foreground">
          Não foi possível carregar a galeria de imagens.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Galeria de Imagens</h2>
          <p className="text-muted-foreground">
            {filteredImages.length} imagens carregadas
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant={viewMode === 'grid' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('grid')}
          >
            <Grid className="h-4 w-4" />
          </Button>
          <Button
            variant={viewMode === 'list' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('list')}
          >
            <List className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Gallery */}
      <AnimatePresence mode="wait">
        {viewMode === 'grid' ? (
          <motion.div
            key="grid"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
          >
            {filteredImages.map((image, index) => (
              <motion.div
                key={image.id}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
              >
                <ImageCard
                  image={image}
                  isSelected={selectedImageId === image.id}
                  onSelect={onImageSelect}
                  showActions={showActions}
                />
              </motion.div>
            ))}
          </motion.div>
        ) : (
          <motion.div
            key="list"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4"
          >
            {filteredImages.map((image, index) => (
              <motion.div
                key={image.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.03 }}
              >
                <ImageListItem
                  image={image}
                  isSelected={selectedImageId === image.id}
                  onSelect={onImageSelect}
                  showActions={showActions}
                />
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {filteredImages.length === 0 && (
        <Card className="p-12 text-center">
          <div className="text-muted-foreground mb-4">
            <Eye className="h-12 w-12 mx-auto" />
          </div>
          <h3 className="text-lg font-medium mb-2">Nenhuma imagem encontrada</h3>
          <p className="text-muted-foreground">
            Faça upload de algumas imagens para começar.
          </p>
        </Card>
      )}
    </div>
  );
}

interface ImageCardProps {
  image: ImageFile;
  isSelected: boolean;
  onSelect?: (image: ImageFile) => void;
  showActions: boolean;
}

function ImageCard({ image, isSelected, onSelect, showActions }: ImageCardProps) {
  return (
    <Card
      className={`overflow-hidden cursor-pointer transition-all duration-300 hover:shadow-lg hover:scale-105 ${
        isSelected ? 'ring-2 ring-primary shadow-lg' : ''
      }`}
      onClick={() => onSelect?.(image)}
    >
      <div className="relative aspect-square overflow-hidden bg-gradient-to-br from-gray-100 to-gray-200">
        <Image
          src={image.url}
          alt={image.name}
          fill
          className="object-cover transition-transform duration-300 hover:scale-110"
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 25vw"
        />
        {isSelected && (
          <div className="absolute inset-0 bg-primary/20 flex items-center justify-center">
            <Badge className="bg-primary text-primary-foreground">
              Selecionada
            </Badge>
          </div>
        )}
      </div>

      <CardContent className="p-4">
        <div className="space-y-2">
          <h3 className="font-medium text-sm truncate" title={image.name}>
            {image.name}
          </h3>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{image.width}×{image.height}</span>
            <span>{(image.size / 1024).toFixed(1)} KB</span>
          </div>
        </div>

        {showActions && (
          <div className="flex gap-2 mt-3">
            <Button size="sm" variant="outline" className="flex-1">
              <Eye className="h-3 w-3 mr-1" />
              Ver
            </Button>
            <Button size="sm" variant="outline">
              <Download className="h-3 w-3" />
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface ImageListItemProps {
  image: ImageFile;
  isSelected: boolean;
  onSelect?: (image: ImageFile) => void;
  showActions: boolean;
}

function ImageListItem({ image, isSelected, onSelect, showActions }: ImageListItemProps) {
  return (
    <Card
      className={`overflow-hidden cursor-pointer transition-all duration-300 hover:shadow-md ${
        isSelected ? 'ring-2 ring-primary shadow-lg' : ''
      }`}
      onClick={() => onSelect?.(image)}
    >
      <CardContent className="p-4">
        <div className="flex gap-4">
          <div className="relative w-20 h-20 rounded-lg overflow-hidden bg-gradient-to-br from-gray-100 to-gray-200 flex-shrink-0">
            <Image
              src={image.url}
              alt={image.name}
              fill
              className="object-cover"
              sizes="80px"
            />
          </div>

          <div className="flex-1 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="font-medium truncate" title={image.name}>
                {image.name}
              </h3>
              {isSelected && (
                <Badge className="bg-primary text-primary-foreground">
                  Selecionada
                </Badge>
              )}
            </div>

            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <span>{image.width}×{image.height}</span>
              <span>{(image.size / 1024).toFixed(1)} KB</span>
              <span>{new Date(image.uploaded_at || '').toLocaleDateString()}</span>
            </div>
          </div>

          {showActions && (
            <div className="flex gap-2 flex-shrink-0">
              <Button size="sm" variant="outline">
                <Eye className="h-4 w-4" />
              </Button>
              <Button size="sm" variant="outline">
                <Download className="h-4 w-4" />
              </Button>
              <Button size="sm" variant="outline">
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
