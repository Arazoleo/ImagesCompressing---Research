using LinearAlgebra
using Colors
using Images
using ImageView

#function qr_algorithm(A; max_iter = 1000, tol = 1e-6)
 #   n = size(A, 1)
 #   Ak = copy(A)
  #  Q_total = I(n) 

   # for i in 1:max_iter
    #    Q, R = qr(Ak)
     #   Ak = R * Q
      #  Q_total *= Q
#
       
 #       off_diagonal = norm(Ak - Diagonal(diag(Ak)))
  #      if off_diagonal < tol
   #         break
    #    end
    #end

    #eigenvalues = diag(Ak)
    #eigenvectors = Q_total
    #return eigenvalues, eigenvectors
#end


function manual_svd(A::Matrix{Float64})
    # Passo 1: Calcular A^T A e AA^T
    AtA = A' * A
    AAt = A * A'

    # Passo 2: Obter autovalores e autovetores
    eig_V = eigen(AtA)
    eig_U = eigen(AAt)

    # Os autovalores são os quadrados dos valores singulares
    S = sqrt.(eig_V.values)

    # Passo 3: Construir a matriz Σ
    Σ = Diagonal(S)

    # Passo 4: Normalizar os autovetores de U e V
    U = eig_U.vectors
    V = eig_V.vectors

    return U, Σ, V
end


function Compress_Image(U, S, V, σ)
    U_σ = U[:, 1:σ]
    S_σ = Diagonal(S[1:σ])
    V_σ = V[:, 1:σ]
    return (U_σ,  S_σ,  V_σ)

end    

function Reconstruct_Image(U_σ,  S_σ,  V_σ)
    return U_σ *  S_σ * V_σ'
end    


image = load("original.png")

gray_esc_img = Gray.(image)
matrix_img_2D = Float64.(gray_esc_img)

println("Memória da imagem original: ", Base.summarysize(gray_esc_img), " bytes")

svd_proc = SingularValues(matrix_img_2D)

σ = 10


U_σ,  S_σ,  V_σ = Compress_Image(svd_proc.U, svd_proc.S, svd_proc.T, σ)

println("Memória de U_σ: ", Base.summarysize(U_σ), " bytes")
println("Memória de S_σ: ", Base.summarysize(Σ_σ), " bytes")
println("Memória de V_σ: ", Base.summarysize(V_σ), " bytes")

compressed_img = Reconstruct_Image(U_σ,  S_σ,  V_σ)

println("Memória da imagem reconstruída: ", Base.summarysize(compressed_img), " bytes")

imshow(compressed_img)

