#=
testConeSquashing.jl
=======
Uses .ConeSquashing to evaluate generalized Jacobian elements of abs-factorable
functions, replicating the results of calculations from the following article:

[1]: KA Khan PI Barton (2012), DOI: 10.1007/978-3-642-30023-3_11
=#

include("../src/ConeSquashing.jl")
using .ConeSquashing

##
println("Running Example 1: A smooth function: \n")

f(x) = (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2
x1 = [2.0, 2.0]

tape1 = record_tape(f, 2, 1)
yJac1, _ = eval_gen_derivative!(tape1, x1)

println("A computation graph of the function: f(x) = (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2 :\n")
display(tape1)

println("A generalized Jacobian element of the function is ", yJac1, ".\n")

##

println("Running Example 2: A non-smooth function of 1 variable: \n")

f2(x) = abs(x[1]) - abs(x[1])
x2 = [0.0]

tape2 = record_tape(f2, 1, 1)
yJac2, _ = eval_gen_derivative!(tape2, x2)

println("A computation graph of the function: f(x) = abs(x[1]) - abs(x[1]) :\n")
display(tape2)

println("The Jacobian matrix of the function is ", yJac2, ".\n")

##

println("Running Example 3: A non-smooth function of 2 variable: \n")

f3(x) = 100.0 - abs(x[1] - x[2]^2)
x3 = [1.0, 2.0]

tape3 = record_tape(f3, 2, 1)
yJac3, _ = eval_gen_derivative!(tape3, x3)

println("A computation graph of the function: f(x) = 100.0 - abs(x[1] - x[2]^2) :\n")
display(tape3)

println("The Jacobian matrix of the function is ", yJac3, ".\n")

##

println("Running Example 4: A non-smooth function using max & min: \n")

f4(x) =  max(min(x[1], -x[2]), x[2] - x[1])

x4 = [0.0, 0.0]

tape4 = record_tape(f4, 2, 1)
yJac4, _ =  eval_gen_derivative!(tape4, x4, verbosity = 1)

println("A computation graph of the function: max(min(x[1], -x[2]), x[2] - x[1]) :\n")
display(tape4)

println("The Jacobian matrix of the function is ", yJac4, ".\n")
