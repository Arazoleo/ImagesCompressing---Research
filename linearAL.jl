using LinearAlgebra
#vetores

function Soma_vetor(v1::Array, v2::Array)
    result_sum = v1 + v2
end

function Produto_escalar(v1::Array, v2::Array)
    product = dot(v1, v2)
end

function Produto_vetorial(v1::Array, v2::Array)
    vetorial = cross(v1,  v2)
end

function Norma_vetor(v1::Array)
    norma = norm(v1)
end


v1 = [1 ,2 ,3]
v2 = [2, 4, 6]

#matrizes

A = [1 2;3 4]
B = [1 0;0 1]

mult = A * B #multiplicação

inversa = inv(A) #inversa

det_A = det(A) #determinante

println(Soma_vetor(v1,v2))
println(Produto_escalar(v1,v2))
println(Produto_misto(v1,v2))
println(Norma_vetor(v1))

println(mult)
println(inversa)
println(det_A)

