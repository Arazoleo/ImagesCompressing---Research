using LinearAlgebra
using Images
using ImageView
using Random
using GLPK
using JuMP

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

function create_matrix_ams(m::Int, n::Int)
    ind = sort(randperm(n)[1:m])
    Phi = zeros(m, n)
    for (j, i) in enumerate(ind)
        Phi[j, i] = 1.0
    end
    return Phi
end


function compressive_sensing_l1(y::Vector{Float64}, Theta::Matrix{Float64})
    m, n = size(Theta)

    model = Model(GLPK.Optimizer)
    set_silent(model) #silenciando output do solver

    @variable(model, s_plus[1:n] >= 0) #variáveis de folga para o problema de minimização
    @variable(model, s_minus[1:n] >= 0)  

    @objective(model, Min, sum(s_plus) + sum(s_minus)) #função objetivo: minimizar a soma dos s_plus e s_minus

    #restrição: Θ(s⁺ - s⁻) = y
    for i in 1:m
        @constraint(model, sum(Theta[i, j] * (s_plus[j] - s_minus[j]) for j in 1:n) == y[i])
    end

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        println("Solver não encontrou solução ótima")
        return zeros(n)
    end
    
    s_star = value.(s_plus) - value.(s_minus)
    
    return s_star
end



function compress_image_cs_l1(img_pth::String, tax::Float64, img_out::String)
    img = load(img_pth)
    img_gray = Gray.(img)
    X = Float64.(img_gray)

    println("Tamanho da imagem: $(size(X))")
    
    m_rows, n_cols = size(X)
    
    m_samples = Int(floor(tax * m_rows))
    println("Amostras por coluna: $m_samples de $m_rows ($(tax * 100)%)")

    Phi = create_matrix_ams(m_samples, m_rows)
    Psi = create_matrix_dct(m_rows)
    Theta = Phi * Psi'

    X_reconstructed = zeros(m_rows, n_cols)

    println("Processando colunas...")
    for j in 1:n_cols
        x_col = X[:, j]

        y = Phi * x_col

        s_recovered = compressive_sensing_l1(y, Theta)

        x_reconstructed = Psi' * s_recovered

        X_reconstructed[:, j] = x_reconstructed

        if j % 50 == 0
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
img_out = "img_reconstructed_l1.jpg"
tax = 0.5

img_reconstructed, error = compress_image_cs_l1(img, tax, img_out)

println("Taxa de amostragem: $(tax * 100)%")
println("Erro MSE: $error")