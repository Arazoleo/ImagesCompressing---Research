# Função para avaliar uma expressão lógica em formato de string usando operadores lógicos de Julia
function avaliar_expressao(expressao::String, valores::Dict{String, Bool})
    # Substituir variáveis pelos valores do dicionário
    for (variavel, valor) in valores
        expressao = replace(expressao, string(variavel) => string(valor))
    end

    # Substituir operadores lógicos pela sintaxe correta de Julia
    expressao = replace(expressao, " AND " => "&&")
    expressao = replace(expressao, " OR " => "||")
    expressao = replace(expressao, "NOT " => "!")
    expressao = replace(expressao, " IMPLICA " => "=>")
    expressao = replace(expressao, " DUPLAIMPLICA " => "==")

    # Imprimir a expressão resultante para verificação
    println("Expressão traduzida para Julia: ", expressao)

    # Avaliar a expressão usando `Meta.parse` e `eval` no contexto local
    try
        # Gerar a expressão juliana para avaliar
        ex = Meta.parse(expressao)

        # Avaliar no contexto local com o dicionário de variáveis
        local_binding = Expr(:block)
        for (variavel, valor) in valores
            push!(local_binding.args, :($variavel = $valor))
        end

        # Usar @eval no contexto local para avaliar
        return eval(local_binding, ex)
    catch e
        println("Erro ao avaliar a expressão: ", e)
        return nothing
    end
end



# Função que verifica se uma expressão lógica satisfaz uma tabela verdade
function verifica_expressao_generica(expressao::String, tabela::Array{Bool, 2})
    n_variaveis = size(tabela, 2) - 1  # Última coluna é o resultado esperado
    resultados = []

    # Iterar sobre cada linha da tabela verdade
    for i in 1:size(tabela, 1)
        # Criar um dicionário com as variáveis e seus valores para a linha atual
        valores = Dict(string("A", j) => tabela[i, j] for j in 1:n_variaveis)
        resultado_esperado = tabela[i, end]

        # Avaliar a expressão com os valores atuais
        resultado_obtido = avaliar_expressao(expressao, valores)

        # Comparar com o resultado esperado e armazenar
        if resultado_obtido === nothing
            println("Erro na linha $i da tabela verdade.")
            return false
        end

        push!(resultados, resultado_obtido == resultado_esperado)
    end

    # Verificar se todos os resultados conferem com a tabela verdade
    return all(resultados)
end


# Exemplo de uso
expressao1 = "A AND (B OR NOT C) IMPLICA D DUPLAIMPLICA E"

# Exemplo de tabela verdade para a expressão (com 5 variáveis)
# Cada linha: valores para A, B, C, D, E seguido do resultado esperado
tabela_verdade = [
    true  true  true  true  true  true;
    true  true  false true  true  true;
    true  false true  false false false;
    false true  false false true true;
    # Adicione mais combinações para testar a expressão completa
]

# Chamada da função para verificar se a expressão corresponde à tabela verdade
resultado = verifica_expressao_generica(expressao1, tabela_verdade)
println("A expressão satisfaz a tabela verdade? ", resultado)
