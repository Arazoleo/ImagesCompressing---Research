using LinearAlgebra
using Images
using ImageView
using CSV
using DataFrames

function SVD_process(A::Matrix{Float64})
    B = A' * A  # AtA
    eig_V = eigen(B)  # extrai os autovalores e autovetores de B
    ε = eig_V.vectors  # autovetores
    α = eig_V.values  # autovalores
    α = abs.(α) #módulo dos autovalores
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
    D = Diagonal(σ) #podemos fazer D utilizando a função Diagonal para uma matriz diagonal

    #Ajusta Σ para o tamanho correto
    Σ = zeros(x, y)
    for i in 1:min(x, y)
        Σ[i, i] = σ[i] #ou podemos fazer a matriz manualmente com um laço for
    end

    # Construção da matriz V
    # Utiliza a fatoração QR para obter uma base ortonormal para o espaço coluna de B (A'*A).
    X = qr(B).Q[:, 1:rank(B)]  # base ortonormal do espaço coluna de B (fatoração QR)
    
    # Para completar a matriz V, concatenamos uma base ortonormal para o espaço nulo de A.
    # Assim, temos uma base completa do espaço de colunas de A', que é o domínio da matriz original A.
    Y = nullspace(A)  # base ortonormal do espaço nulo de A
    V = hcat(X, Y)  # Matriz V é formada concatenando X e Y

    # Construção da matriz U a partir da matriz V
    if x <= y
        U = zeros(x, size(Σ, 1))
        for i in 1:size(Σ, 1)
            if i <= size(V, 2)
                # U é calculado usando a definição de SVD: U[:, i] = (1 / σ[i]) * A * V[:, i]
                # Onde A*V[:, i] nos dá a componente da matriz original na direção de V[:, i].
                # Dividimos pelo valor singular correspondente para normalizar essa direção.
                U[:, i] = (1 / σ[i]) * A * V[:, i]
            end
        end
    else
        # Caso onde o número de linhas é maior que o número de colunas
        # Inicialmente, construímos uma matriz W que guarda os componentes das colunas ortogonais de U.
        W = zeros(x, y)
        for i in 1:y
            if i <= size(V, 2)
                # Similar ao caso anterior, construímos as primeiras y colunas de U.
                W[:, i] = (1 / σ[i]) * A * V[:, i]
            end
        end
        # Para completar a matriz U, concatenamos uma base ortonormal para o espaço nulo de A'.
        # Essa é a parte complementar que completa a base ortonormal do espaço das colunas de A.
        Y = nullspace(A')
        U = hcat(W, Y)
    end

    return U, Σ, V
end


# Carrega e converte a imagem para escala de cinza
A = load("raio.jpg")
B = Gray.(A)
C = Float64.(B)

# Tamanho da imagem original
println("Tamanho da imagem original: ", size(C))

# Decomposição SVD manual
U, Σ, V = SVD_process(C)

# Tamanho das matrizes U, Σ e V
println("Tamanho da matriz U: ", size(U))
println("Tamanho da matriz Σ: ", size(Σ))
println("Tamanho da matriz V: ", size(V))



k = 200 # Defina o valor de k aqui

# Verifica se k não excede o número de valores singulares disponíveis
k = min(k, size(Σ, 1))


# Reconstrução da imagem com k valores singulares
global A_approx = zeros(size(C))
for i in 1:k
    global A_approx += Σ[i, i] * U[:, i] * V[:, i]'
end

CSV.write("U.csv", DataFrame(U, :auto))
CSV.write("Sigma.csv", DataFrame(Σ, :auto))
CSV.write("V.csv", DataFrame(V, :auto))

# Tamanho da imagem comprimida
println("Tamanho da imagem comprimida: ", size(A_approx))

# Visualiza a imagem comprimida
imshow(A_approx)

# Salva a imagem comprimida
save("img_compressed_200.jpeg", A_approx)
