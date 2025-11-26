using Images
using Colors
using Statistics
using LinearAlgebra



image = load("sky.png")

pixels_mat = Float64.(channelview(image))

height, width, canals = size(pixels_mat)

println("Dimensões: $width x $height")
println("\nQuantidade de Canais: $canals")


#println(pixels_mat) printa a matriz (imensa, não recomendo printá-la)

#eigenvalues_A, eigenvectors_A = eigen(pixels_mat) matriz RGB não numérica - método inválido

