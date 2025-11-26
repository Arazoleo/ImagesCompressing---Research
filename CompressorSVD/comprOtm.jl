using LinearAlgebra
using Images
using ImageView

function SVD_process(A::Matrix{Float64})
    B = A' * A  # AtA
    eig_V = eigen(B)  # extrai os autovalores e autovetores de B
    ε = eig_V.vectors  # autovetores
    α = eig_V.values  # autovalores

    # Construção da matriz S
    m = length(α)
    for i in 1:m
        if α[i] < exp(-10)  # restrição para evitar raízes negativas por erro de arredondamento
            α[i] = 0
        end
    end

    α = sort(α, rev=true)  # colocar em ordem decrescente
    σ = sqrt.(α)  # aplicar a √ para encontrar os valores singulares

    x, y = size(A)
    if x < y
        while length(σ) > x
            if σ[end] == 0
                pop!(σ)
            end
        end
    end

    D = Diagonal(σ)  # formação da matriz diagonal
    p, q = size(D)
    if p < x
        Σ = vcat(D, zeros(x - p, y))
    elseif q < y
        Σ = hcat(D, zeros(x, y - q))
    else
        Σ = D
    end

    # Construção da matriz V
    X = qr(B).Q[:, 1:rank(B)]  # base ortonormal do espaço coluna de B
    Y = nullspace(A)  # base ortonormal do espaço nulo de A
    V = hcat(X, Y)

    # Construção da matriz U a partir da V
    if x <= y
        U = zeros(x, x)
        for i in 1:x
            U[:, i] = (1 / Σ[i, i]) * A * V[:, i]
        end
    else
        W = zeros(x, y)
        for i in 1:y
            W[:, i] = (1 / Σ[i, i]) * A * V[:, i]
        end
        Y = nullspace(A')
        U = hcat(W, Y)
    end

    return U, Σ, V
end

# Carrega e converte a imagem para escala de cinza
A = load("paisa.jpeg")
B = Gray.(A)
C = Float64.(B)

# Tamanho da imagem original
original_size = sizeof(C)
println("Tamanho da imagem original: ", original_size, " bytes")

# Decomposição SVD manual
U, Σ, V = SVD_process(C)

# Solicita o valor de k (número de valores singulares)
k = 100  # Defina o valor de k aqui

# Verifica se k não excede o número de valores singulares disponíveis
k = min(k, size(Σ, 1))

# Reconstroi a imagem com k valores singulares
A_approx = zeros(size(C))
for i in 1:k
    A_approx += Σ[i, i] * U[:, i] * V[:, i]'
end

# Tamanho da imagem comprimida
compressed_size = sizeof(A_approx)
println("Tamanho da imagem comprimida: ", compressed_size, " bytes")

# Visualiza a imagem comprimida
imshow(A_approx)

# Salva a imagem comprimida
save("pai_compressed.jpeg", A_approx)
