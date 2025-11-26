using LinearAlgebra

A = [1.0 0.0 ; 1.0 3.0]
B = [2.0 1.0 ; 0.0 1.0]

eigens_generalized = eigen(A, B)

eigenval_genera = eigens_generalized.values
eigenvec_genera = eigens_generalized.vectors


println("Os autovalores generalizados são: ")
println(eigenval_genera)
println("\n\nOs autovetores generalizados são: ")
println(eigenvec_genera)

