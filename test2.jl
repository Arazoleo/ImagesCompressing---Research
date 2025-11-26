using Images, Colors, Statistics, LinearAlgebra, ImageView

# Carregar a imagem e convertê-la para tons de cinza
image = load("raio.jpg")
gray_image = Gray.(image)
imshow(gray_image)

# Converter a imagem para um array de pixels
pixels_array = Float64.(channelview(gray_image))

# Calcular a média dos pixels
mean_pixel = mean(pixels_array, dims=2)

# Centralizar os dados
centered_pixels_array = pixels_array .- mean_pixel

# Calcular a matriz de covariância
cov_matrix = centered_pixels_array * centered_pixels_array' / (size(centered_pixels_array, 2) - 1)

# Calcular autovalores e autovetores
eigenvalues, eigenvectors = eigen(cov_matrix)

# Selecionar os k componentes principais
k = 4000
principal_components = eigenvectors[:, 1:k]

# Projetar os dados nos componentes principais
projected_data = principal_components' * centered_pixels_array

# Para descompressão, reconstruir a imagem
reconstructed_data = principal_components * projected_data .+ mean_pixel

# Converter o array de volta para a imagem
reconstructed_image = Gray.(reshape(reconstructed_data, size(gray_image)))
imshow(reconstructed_image)
