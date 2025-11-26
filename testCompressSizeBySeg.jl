using Images, FileIO, ImageView

# Carregar a imagem
image = load("terra.jpg")

# Converter a imagem para escala de cinza
grayIM = Gray.(image)

# Mostrar a imagem em escala de cinza
imshow(grayIM)

# Inicializar uma imagem para a segmentação
seg_im = similar(grayIM)

# Limiar para segmentação
thr = 0.2

# Realizar a segmentação baseada na diferença de pixels
for i in 2:size(grayIM, 1) - 1
    for j in 2:size(grayIM, 2) - 1
        if abs(grayIM[i, j] - grayIM[i, j + 1]) < thr
            seg_im[i, j] = grayIM[i, j]
        else
            seg_im[i, j] = 1.0
        end        
    end
end    

# Mostrar a imagem segmentada
imshow(seg_im)

origina_mem = Base.summarysize(grayIM)
seg_mem = Base.summarysize(seg_im)

println("Original: " ,origina_mem)
println("Segmentada: ", seg_mem)
# Salvar as imagens em arquivos
save("original.png", grayIM)
save("segmented.png", seg_im)

# Obter os tamanhos dos arquivos
original_file_size = filesize("original.png")
segmented_file_size = filesize("segmented.png")

println("Original: ", original_file_size, " bytes")
println("Segmentada: ", segmented_file_size, " bytes")

