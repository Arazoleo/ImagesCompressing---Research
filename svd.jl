using Images
using LinearAlgebra
using ImageView
using Colors

# Carregar a imagem
img = load("original.png")

# Converter a imagem para escala de cinza
img_gray = Gray.(img)

# Converter a imagem para uma matriz de floats
matrix = Float64.(img_gray)

# Realizar a Decomposição em Valores Singulares (SVD)
U, S, V = svd(matrix)

# Escolha o número de componentes principais
k = 100  # Mantenha os 50 maiores valores singulares, por exemplo

# Truncar as matrizes U, S, V
U_k = U[:, 1:k]
S_k = Diagonal(S[1:k])
V_k = V[:, 1:k]

# Reconstruir a imagem comprimida
compressed_matrix = U_k * S_k * V_k'

# Converter a matriz comprimida de volta para o tipo Gray
compressed_img_gray = Gray.(compressed_matrix)

# Visualizar a imagem comprimida
imshow(compressed_img_gray)

# Salvar a imagem comprimida
save("compressed.png", compressed_img_gray)