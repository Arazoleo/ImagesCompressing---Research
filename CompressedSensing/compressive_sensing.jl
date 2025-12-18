using LinearAlgebra
using Images
using ImageView
using Random
using FFTW

function gc(A::Matrix{Float64}, b::Vector{Float64}; tol = 1e-8, max_iter = 5000)
    n = length(b)
    x = zeros(n) # sol inicial
    r = b - A * x # resíduo inicial 
    p = copy(r) # direção de busca
    rs_old = dot(r, r)  # r'r

    for i in 1:max_iter
        Ap = A * p
        alpha = rs_old / dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = dot(r, r)
        if sqrt(rs_new) < tol
            println("Gradientes Conjugados convergiu em $i iterações")
            return x
        end
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    end
    println("Gradientes Conjugados: máximo de iterações atingido")
    return x
end

function create_matrix_ams(m::Int, n::Int)
    ind = sort(randperm(n)[1:m])
    Phi = zeros(m, n)
    for (j, i) in enumerate(ind)
        Phi[j, i] = 1.0
    end
    return Phi
end

function create_matrix_dct(n::Int)
    Psi = zeros(n, n)
    for k in 0:(n-1)
        for j in 0:(n-1)
            if k == 0
                Psi[k+1, j+1] = 1.0 / sqrt(n)
            else
                Psi[k+1, j+1] = sqrt(2.0 / n) * cos(π * k * (2j + 1) / (2n))
            end
        end
    end
    return Psi
end


function compressive_sensing_l2(y::Vector{Float64}, Theta::Matrix{Float64})
    A = Theta * Theta'
    
    lambda = 1e-6
    A_reg = A + lambda * I
    
    b = gc(A_reg, y)
    s_star = Theta' * b
    return s_star
end


function compress_image_cs(img_pth::String, tax::Float64, img_out::String)
    img = load(img_pth)
    img_gray = Gray.(img)
    X = Float64.(img_gray)


    println("Tamanho da imagem: $(size(X))")
    
    m_rows, n_cols = size(X)

    m_samples = Int(floor(tax * m_rows))
    println("Amostras por coluna: $m_samples de $m_rows ($(tax * 100)%)")

    Phi = create_matrix_ams(m_samples, m_rows) # matriz de amostragem
    Psi = create_matrix_dct(m_rows) # base DCT

    Theta = Phi * Psi' # matriz de medição

    X_reconstructed = zeros(m_rows, n_cols)

    println("Processando colunas...")
    for j in 1:n_cols
        x_col = X[:, j]

        y = Phi * x_col

        s_recovered = compressive_sensing_l2(y, Theta)

        x_reconstructed = Psi' * s_recovered

        X_reconstructed[:, j] = x_reconstructed

        if j % 100 == 0
            println("Coluna $j de $n_cols processada")
        end
    end

    X_reconstructed = replace(X_reconstructed, NaN => 0.0)
    X_reconstructed = clamp.(X_reconstructed, 0.0, 1.0)

    println("Tamanho da imagem reconstruída: ", size(X_reconstructed))

    mse = sum((X .- X_reconstructed).^2) / (m_rows * n_cols)
    println("Erro MSE: $mse")

    save(img_out, X_reconstructed)
    println("Imagem salva em: $img_out")

    imshow(X_reconstructed)

    return X_reconstructed, mse
end





img = "img.jpg"  
img_r = "img_reconstructed.jpg"
tax = 0.7
img_reconstructed, error = compress_image_cs(img, tax, img_r)

println("Taxa de amostragem: $(tax * 100)%")
println("Erro MSE: $error")