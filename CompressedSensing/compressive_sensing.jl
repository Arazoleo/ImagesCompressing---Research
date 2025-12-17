using LinearAlgebra
using Images
using ImageView
using Random
using FFTW


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

    X_reconstructed = clamp.(X_reconstructed, 0.0, 1.0)

    println("Tamanho da imagem reconstruída: ", size(X_reconstruida))

    mse = sum((X .- X_reconstruida).^2) / (m_rows * n_cols)
    println("Erro MSE: $mse")

    save(img_out, X_reconstructed)
    println("Imagem salva em: $output_path")

    imshow(X_reconstructed)

    return X_reconstructed, mse
end





tax = 0.5
img_reconstructed, error = compress_image_cs(img, tax, img_r)

println("Taxa de amostragem: $(tax * 100)%")
println("Erro MSE: $error")