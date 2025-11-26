using LinearAlgebra
using Images
using ImageView
using CSV
using DataFrames

function SVD_fac(A::Matrix{Float64})
    B = A' * A # A^t A 
    eig_V = eigen(B) # extração dos autovalores e autovetores de B
    ϵ = eig_V.vectors # autovetores
    α = eig_V.values # autovalores
    α = abs.(α) #módulo dos autovalores

    # Composição de S (Matriz)

    m = length(α)
    for i in 1:m
        if α[i] < exp(-10) # restrição para evitar números complexos
            α[i] = 0.0
        end
    end

    α = sort(α, rev=true) # em ordem decrescente
    σ = sqrt.(α) # aplicação da raíz quadrada para encontrar os valores singulares

    x, y = size(A)
    D = Diagonal(σ) # assim, fazendo D utilizando a função Diagonal para uma matriz diagonal

    Σ = zeros(x, y) # ajustando o tamanho de Σ
    for i in 1:min(x, y)
        Σ[i, i] = σ[i]
    end

    # Composição da matriz V
    # Utiliza a fatoração QR para obter uma base ortonormal para o espaço coluna de B (A'*A).
    X = qr(B).Q[:, 1:rank(B)] # base ortonormal do espaço coluna de B (fatoração QR)

    # Para completar a matriz V, concatenamos uma base ortonormal para o espaço nulo de A.
    # Assim, temos uma base completa do espaço de colunas de A', que é o domínio da matriz original A.
    Y = nullspace(A) #base ortonormal do espaço nulo de a
    V = hcat(X, Y) # matriz V é formada através da concatenação de X e Y
    
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
                #similar ao caso anterior, construímos as primeiras y colunas de U.
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

    
function compress_rgb_image(image_path::String, k::Int, output_path::String)
    # Carrega a imagem
    A = load(image_path)
    
    # Converte para RGB se necessário
    A_rgb = RGB.(A)
    
    # Extrai os canais R, G, B separadamente
    red_channel = Float64.(red.(A_rgb))
    green_channel = Float64.(green.(A_rgb))
    blue_channel = Float64.(blue.(A_rgb))
    
    println("Tamanho da imagem original: ", size(red_channel))
    
    # Aplica SVD em cada canal separadamente
    println("Processando canal vermelho...")
    U_r, Σ_r, V_r = SVD_fac(red_channel)
    
    println("Processando canal verde...")
    U_g, Σ_g, V_g = SVD_fac(green_channel)
    
    println("Processando canal azul...")
    U_b, Σ_b, V_b = SVD_fac(blue_channel)
    
    # Verifica se k não excede o número de valores singulares disponíveis
    k_red = min(k, size(Σ_r, 1))
    k_green = min(k, size(Σ_g, 1))
    k_blue = min(k, size(Σ_b, 1))
    
    # Reconstrução de cada canal com k valores singulares
    red_approx = zeros(size(red_channel))
    for i in 1:k_red
        red_approx += Σ_r[i, i] * U_r[:, i] * V_r[:, i]'
    end
    
    green_approx = zeros(size(green_channel))
    for i in 1:k_green
        green_approx += Σ_g[i, i] * U_g[:, i] * V_g[:, i]'
    end
    
    blue_approx = zeros(size(blue_channel))
    for i in 1:k_blue
        blue_approx += Σ_b[i, i] * U_b[:, i] * V_b[:, i]'
    end
    
    # Garante que os valores estão no intervalo [0, 1]
    red_approx = clamp.(red_approx, 0.0, 1.0)
    green_approx = clamp.(green_approx, 0.0, 1.0)
    blue_approx = clamp.(blue_approx, 0.0, 1.0)
    
    # Reconstrói a imagem RGB
    height, width = size(red_channel)
    A_approx = Array{RGB{Float64}}(undef, height, width)
    
    for i in 1:height
        for j in 1:width
            A_approx[i, j] = RGB(red_approx[i, j], green_approx[i, j], blue_approx[i, j])
        end
    end
    
    # Salva as matrizes SVD para cada canal (opcional)
    # CSV.write("U_red.csv", DataFrame(U_r, :auto))
    # CSV.write("Sigma_red.csv", DataFrame(Σ_r, :auto))
    # CSV.write("V_red.csv", DataFrame(V_r, :auto))
    
    # CSV.write("U_green.csv", DataFrame(U_g, :auto))
    # CSV.write("Sigma_green.csv", DataFrame(Σ_g, :auto))
    # CSV.write("V_green.csv", DataFrame(V_g, :auto))
    
    # CSV.write("U_blue.csv", DataFrame(U_b, :auto))
    # CSV.write("Sigma_blue.csv", DataFrame(Σ_b, :auto))
    # CSV.write("V_blue.csv", DataFrame(V_b, :auto))
    
    println("Tamanho da imagem comprimida: ", size(A_approx))
    
    # Visualiza a imagem comprimida
    imshow(A_approx)
    
    # Salva a imagem comprimida
    save(output_path, A_approx)
    
    return A_approx, (U_r, Σ_r, V_r), (U_g, Σ_g, V_g), (U_b, Σ_b, V_b)
end

# Uso da função
k = 50 # Número de valores singulares para manter
compressed_image, svd_red, svd_green, svd_blue = compress_rgb_image("paisa.jpeg", k, "paisa_compressed.jpeg")

# Para comparar com a imagem original
original = load("paisa.jpeg")

imshow(compressed_image)  # Mostra a comprimida



