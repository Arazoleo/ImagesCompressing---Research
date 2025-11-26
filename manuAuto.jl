using LinearAlgebra

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


A = [1.0 0.0; 1.0 3.0]
eigenvalues, eigenvectors = qr_algorithm(A)

println("Autovalores: ", eigenvalues)
println("Autovetores: ", eigenvectors)
