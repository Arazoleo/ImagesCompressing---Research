using Images

imwidth = 800
imheight = 600

img = Array{RGB}(undef, imheight, imwidth)

for i in 1:imheight
    for j in 1:imwidth
        r = (i - 1)/(imheight - 1)
        g = 1.0 - (j - 1)/(imwidth - 1)
        b = 0.75

        img[i, j] = RGB(r, g, b)
    end
end       

save("renderized/imagem0.png", img)
