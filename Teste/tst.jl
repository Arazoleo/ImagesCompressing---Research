using LinearAlgebra

struct Ray
    origin::Vector{Float64}
    direction::Vector{Float64}
end


function point_at_parameter(ray::Ray, t::Float64)
    return ray.origin + t * ray.direction
end


function color(ray::Ray)
    unit_direction = normalize(ray.direction)
    t = 0.5 * (unit_direction[2] + 1.0)
    return (1.0 - t) * [1.0, 1.0, 1.0] + t * [0.5, 0.7, 1.0]
end


width = 800
height = 600


image = zeros(RGB, width, height)


lower_left_corner = [-2.0, -1.0, -1.0]
horizontal = [4.0, 0.0, 0.0]
vertical = [0.0, 2.0, 0.0]
origin = [0.0, 0.0, 0.0]


for j in 1:height
    for i in 1:width
        u = i / width
        v = j / height
        direction = lower_left_corner + u * horizontal + v * vertical - origin
        ray = Ray(origin, direction)
        col = color(ray)
        image[i, j] = RGB(col[1], col[2], col[3])
    end
end

# Salvando a imagem
save("empty_background.png", image)
