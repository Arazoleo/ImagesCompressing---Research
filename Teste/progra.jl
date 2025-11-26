function Calc(a, b, op)
    if op == "+"
        return a + b
    elseif op == "-"
        return a - b
    elseif op == "*"
        return a * b
    elseif op == "/"
        return a / b
    end
end           

while true
    println("Digite um número: ")
    a = readline()
    num = parse(Int, a)
    println("Digite um número: ")
    b = readline()
    num2 = parse(Int, b)
    println("Digite uma operação: ")
    o = readline()

    resultado = Calc(num, num2, o)
    println("O resultado é : " * string(resultado))

    println("Deseja sair? ")
    resp = readline()

    if resp == "s"
        break
    end
end

