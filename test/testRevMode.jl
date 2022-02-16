# test ReverseAD.jl

include("../src/ReverseAD.jl")

using .ReverseAD

## Example 1: the Rosenbrock function
##   from Example 2.2 of Naumann (2012), DOI:10.1137/1.9781611972078

# construct AD tape for function
f1(x) = (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2
tape1 = generate_tape(f1, 2, 1)

# carry out reverse AD mode on tape
x1 = [2.0, 2.0]
yBar1 = [1.0]
y1, xBar1 = reverse_AD!(tape1, x1, yBar1)

println("For Example 1:")
println("  Function tape:")
println(tape1)
@show y1
@show xBar1

## Example 2: the extended Rosenbrock function
##   from Section 1.4.3 of Naumann (2012), DOI:10.1137/1.9781611972078

# construct AD tape for function
function g2(x, n)
    y = 0.0
    for i in 1:(n-1)
        y += (1.0 - x[i])^2
        y += 10.0 * (x[i+1] - x[i]^2)^2
    end
    return y
end

n = 10             # domain dimension
f2(x) = g2(x, n)   # fix n at 10
tape2 = generate_tape(f2, n, 1)

# carry out reverse AD mode on tape
x2 = 2.0*ones(n)
yBar2 = [1.0]
y2, xBar2 = reverse_AD!(tape2, x2, yBar2)

println("For Example 2:")
println("  Function tape:")
println(tape2)
@show y2
@show xBar2


