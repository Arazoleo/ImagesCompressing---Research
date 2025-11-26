using Images
using Colors
using Statistics
using LinearAlgebra
using ImageView




function qr_algorithm(A; max_iter = 1000, tol = 1e-6)
    n = size(A, 1)
    Ak = copy(A)
    Q_total = I(n) 

    for i in 1:max_iter
        Q, R = qr(Ak)
        Ak = R * Q
        Q_total *= Q

       
        off_diagonal = norm(Ak - Diagonal(diag(Ak)))
        if off_diagonal < tol
            break
        end
    end

    eigenvalues = diag(Ak)
    eigenvectors = Q_total
    return eigenvalues, eigenvectors
end


image = load("raio.jpg")

gray_image = Gray.(image)
#imshow(gray_image)

pixels_array = Float64.(channelview(gray_image))

mean_pixel = mean(pixels_array, dims=2)

centered_pixels_array = pixels_array .- mean_pixel


cov_matrix = centered_pixels_array * centered_pixels_array' / (size(centered_pixels_array, 2) - 1)


eigenvalues, eigenvectors = qr_algorithm(cov_matrix)


println("Autovalores: ", eigenvalues )
println("\n\nAutovetores: ", eigenvectors)

