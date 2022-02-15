# test ReverseAD.jl

include("ReverseAD.jl")

using .ReverseAD

# construct AD tape for provided function
f(x) = [x[1] - x[2]]
fDomainDim = 3
fRangeDim = 1
tape = generate_tape(f, fDomainDim, fRangeDim)

# carry out reverse AD mode on tape
x = [1.0, 3.0, 2.0]
yBar = [2.0]
y, xBar = reverse_AD!(tape, x, yBar)

println(tape)
@show y
@show xBar
