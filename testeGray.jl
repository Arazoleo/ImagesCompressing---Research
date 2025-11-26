using Images
using Colors
using Statistics
using LinearAlgebra
using ImageView


image = load("raio.jpg")

gray_image = Gray.(image)
imshow(gray_image)

pixels_array = Float64.(channelview(gray_image))

mean_pixel = mean(pixels_array, dims=2)

centered_pixels_array = pixels_array .- mean_pixel


cov_matrix = centered_pixels_array * centered_pixels_array' / (size(centered_pixels_array, 2) - 1)


eigenvalues, eigenvectors = eigen(cov_matrix)




