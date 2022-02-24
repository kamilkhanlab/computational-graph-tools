# test ReverseAD.jl

include("../src/ReverseAD.jl")

using .ReverseAD

## Example 1: the Rosenbrock function
##   from Example 2.2 of Naumann (2012), DOI:10.1137/1.9781611972078

# construct AD tape for function
tape1 = record_tape(x -> (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2, 2, 1)

# carry out reverse AD mode on tape
x1 = [2.0, 2.0]
yBar1 = [1.0]
y1, xBar1 = reverse_AD!(tape1, x1, yBar1)

println("For Example 1:")
println("  Function tape:")
println(tape1)
println("  At x = ", x1, ":")
println("    f(x) = ", y1)
println("    gradient of f at x = ", xBar1)

## Example 2: the extended Rosenbrock function
##   from Section 1.4.3 of Naumann (2012), DOI:10.1137/1.9781611972078

# construct AD tape for function
n = 10    # domain dimension
tape2 = record_tape(n, 1) do x
    y = 0.0
    for i in 1:(n-1)
        y += (1.0 - x[i])^2
        y += 10.0 * (x[i+1] - x[i]^2)^2
    end
    return y
end

# carry out reverse AD mode on tape
x2 = 2.0*ones(n)
yBar2 = [1.0]
y2, xBar2 = reverse_AD!(tape2, x2, yBar2)

println("\nFor Example 2:")
println("  Function tape:")
println(tape2)
println("  At x = ", x2, ":")
println("    f(x) = ", y2)
println("    gradient of f at x = ", xBar2)

# generate reverse AD code in MATLAB
generate_revAD_matlab_code!(tape1)
println("\nMATLAB reverse AD code generated for Example 1 as fRevAD.m")
