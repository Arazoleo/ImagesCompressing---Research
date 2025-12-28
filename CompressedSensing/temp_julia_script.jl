
using Images
using LinearAlgebra
using JuMP
using GLPK
using Random

include("/Users/arazoleonardo/ImagesCompressing---Research/CompressedSensing/compressive_sensing_l1.jl")

img = "temp_input.jpg"
img_out = "dataset/cs_reconstructed/108082.jpg"
tax = 0.5

img_reconstructed, error = compress_image_cs_l1(img, tax, img_out)
println("Processado: 108082.jpg, MSE: ", error)
