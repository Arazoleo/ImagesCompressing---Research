struct Vec3{T}
    x::T
    y::T
    z::T
end #estamos dando o nome Vec3 para o tipo Array, semelhante a um typedef, estamos tamném dizendo que será um vetor derivado de números flutuantes

3=function Vec3(x::T, y::T, z::T) where T
    [x, y, z]
end


function norm(v::Vec3)
    sqrt(sum(map(x->x^2, v)))
end

function Prod_Esc(v1::Vec3, v2::Vec3)
    sum(v1 .* v2) #sintaxe do .* indica produto componente a componente
    
end

function Unitary_Vector(v::Vec3)
    v / norm(v)
end    

#a = Vec3(2.0, 1.0 ,2.0)
#b = Vec3(2.0, 4.0, 3.0)
#norma = norm(a)
#escalar = Prod_Esc(a, b)
#println("O vetor a é: ",  a)
#println("A norma de a ", norma)
#println(escalar)

struct Ray{T <: AbstractFloat}
    origem::Vec3{T}
    direcao::Vec3{T}

    function Ray{T}(org::Vec3{T}, dir::Vec3{T}) where T
        new(org, Unitary_Vector(dir))
    end
end    

function Ray(org::Vec3{T} , dir::Vec3{T}) where T
    Ray{T}(org, dir)
end    



