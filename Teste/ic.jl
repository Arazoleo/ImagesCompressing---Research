using Images
include("array.jl")
 
# imagem
aspectratio = 16 / 9
imWidth = 800
imHeight = trunc(Int64, imWidth / aspectratio)
img = RGB.(zeros(imHeight, imWidth))

# camera
view_height = 2.0
view_width = view_height * aspectratio
hor = Vec3(view_width, 0.0, 0.0)
ver = Vec3(0.0, view_height, 0.0)
focal = 1.0
origem = Vec3(0.0, 0.0, 0.0)
cant_inf_esq = origem - hor/2.0 - ver/2.0 - Vec3(0.0, 0.0, 1.0)

println("Imagem tamanho: $imWidth x $imHeight")

function backgroundcolor(dir)
    t = 0.5 * (dir[2] + 1.0) 
    (1 - t)RGB(1.0, 1.0, 1.0) + t*RGB(0.5, 0.7, 1.0) 
end

function Cor_raio(ray::Ray)
    backgroundcolor(ray.direcao)
end

for j = 1:imHeight
    for i = 1:imWidth
        u = (i - 1)/(imWidth - 1)
        v = 1.0 - (j - 1)/(imHeight - 1)
        dir = cant_inf_esq + u*hor + v*ver - origem
        raio = Ray(origem, dir)
        img[j, i] = Cor_raio(raio)  
    end
end

save("renderized/imagem2.png", img)

