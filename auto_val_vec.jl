using LinearAlgebra


A = [1.0 0.0 ; 1.0 3.0]

# Calculando autovalores e autovetores
eigens = eigen(A)

# Acessando autovalores e autovetores
eigenvalues_A = eigens.values
eigenvectors_A = eigens.vectors

# Exibindo os resultados
println("Os autovalores são: ")
println(eigenvalues_A)
println("\nOs autovetores são: ")
println(eigenvectors_A)
