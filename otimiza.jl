using JuMP
using GLPK

modelo = Model(GLPK.Optimizer)

@variable(modelo, 0 <= x <= 10)

@objective(modelo, Min, x + 2)

@constraint(modelo, x >= 1)

optimize!(modelo)

solucao = termination_status(modelo)
if solucao == MOI.OPTIMAL
    println("Solução ótima existe!")
    println("Temos x : ", value(x))
    println("Temos o valor da função objetivo: ", objective_value(modelo))
else
    println("Solução ótima não existe!")
end