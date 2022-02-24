# computational-graph-tools
A Julia module for automatically constructing the computational graph/tape of a supplied composite function, and an implementation of the reverse mode of automatic differentiation (AD) that operates on this graph to either compute adjoint derivatives or to generate MATLAB code that computes adjoint derivatives. These implementations are designed to be relatively straightforward to understand and adapt, and  without depending on any packages external to Julia.

Tested in Julia v1.4.1.

## Computational graph generation
The module `CompGraphs` in [CompGraphs.jl](src/CompGraphs.jl) is a facility for building the computational graph for a composite function.

### What's a computational graph?
A computational graph represents a composite function as its individual elemental operations, keeping track of when the output of one operation is an input of another. For example, consider the following composite function (from Example 2.2 of Naumann (2012)):
```julia
f(x) = (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2
```
This function is not intrinsic to Julia, but does depend on various elemental "scientific calculator" operations that Julia understands. We could reformulate `f` to explicitly consider these elemental operations one by one in a recipe:
```julia
function f(x)
  v = zeros(12)

  # load inputs
  v[1] = x[1]
  v[2] = x[2]

  # evaluate elemental operations one by one
  v[3] = 1.0
  v[4] = v[3] - v[1]
  v[5] = v[4]^2
  v[6] = v[1]^2
  v[7] = v[2] - v[6]
  v[8] = v[7]^2
  v[9] = 100.0
  v[10] = v[9]*v[8]
  v[11] = v[5] + v[10]
  v[12] = v[11]

  # recover output
  return v[12]
end
```
This is the same function `f` as before. A computational graph of `f` is a directed acyclic graph with one node for each intermediate quantity `v[i]` in this representation. Each node `v[i]` "knows" its index `i` and the mathematical operation that it corresponds to, and is connected by an edge to each previous node `v[j]` that was an input of this operation.

### Implementation overview
The module `CompGraphs` in [CompGraphs.jl](src/CompGraphs.jl) exports the definitions of two parametric structs: `CompGraph{T, P}` and `GraphNode{P}`. A `CompGraph` is intended to hold the computational graph of a single composite function, and is made up of `GraphNode`s. The parametric types `T` and `P` are intended to hold application-specific data, respectively pertaining to the overall graph and particular nodes. 

#### Fields of `GraphNode{P}`:

- `operation::Symbol`: the mathematical operation described by the node (e.g. `:+`). The supported operations are listed in `unaryOpList` and `binaryOpList` in [CompGraphs.jl](src/CompGraphs.jl).
- `parentIndices::Vector{Int}`: list of indices of parent nodes, representing inputs of this node's `operation`.
- `constValue::Union{Float64, Int}`: holds operation-specific constants such as the constant integer exponent `3` in the cube operation `f(x) = x^3`. Most operations don't use this field.
- `data::P`: any application-specific data or computational results pertaining to this particular node. The parameter `P` may represent e.g. a user-defined mutable struct.

#### Fields of `CompGraph{T, P}`:

- `nodeList::Vector{GraphNode{P}}`: holds the nodes of the graph.
- `domainDim::Int`: domain dimension of the represented function.
- `rangeDim::Int`: range dimension ofo the represented function.
- `data::T`: any application-specific data or results that are not specific to one particular node.

#### Exported functions:

- `load_function!(f::Function, graph::CompGraph{T, P}, initP::P) where {T, P}`: 
  -   loads the composite function `f` into `graph`, which must be constructed in advance. Each resulting `GraphNode`'s `data` field is initialized with the value `deepcopy(initP)`. The computational graph is generated using operator overloading, passing in an internal `GraphBuilder` object in place of any `Float64`. So, `f` must be written as if it takes a `Vector{Float64}` input, and returns either a `Float64` or `Vector{Float64}` output, but without actually specifying that these input/outputs are `Float64`s.

- `is_function_loaded(graph::CompGraph)::Bool`: 
  - checks if a composite function has already been loaded into `graph`, by confirming that `graph.nodeList` is nonempty.

The following constructor for `CompGraph` is available in addition to the usual default constructor: 

- `CompGraph{T, P}(domainDim::Int, rangeDim::Int) where {T, P}`: requires that a constructor `T()` (with no arguments) is available to provide a value for `this.data`. 

## Reverse AD mode implementation
The module `ReverseAD` in [ReverseAD.jl](src/ReverseAD.jl) contains a straightforward implementation of the standard reverse mode of automatic differentiation (AD) for smooth functions. This implementation operates on a `CompGraph` and requires `CompGraphs.jl` in the same directory. It is intended to show how `CompGraphs.jl` may be used for numerical computation. 

### Overview
The features of the `ReverseAD` module are illustrated in [testRevAD.jl](test/testRevAD.jl). The module exports the following functions:

- `tape = record_tape(f::Function, domainDim::Int, rangeDim::Int)`: 

  - returns an AD-amenable "tape" for the provided function `f`, which is secretly just a specialized `CompGraph`. The domain and range dimensions must be provided, and `f` must be of the format required by `CompGraphs::load_function!`. 

- `(y, xBar) = reverse_AD!(tape::CompGraph, x::Vector{Float64}, yBar::Vector{Float64})`: 

  - performs the reverse mode of AD on the output `tape` of `record_tape`. With `f` denoting the recorded function, this returns `y = f(x)` and `xBar = (Df(x))'*yBar`. In particular, if `f` is scalar-valued and `yBar = [1.0]`, then `xBar` is the gradient of `f` at `x`. The `data` fields of `tape` and its nodes are used to hold results of intermediate computations.

- `generate_revAD_matlab_code!(tape::CompGraph, funcName::AbstractString = "f", fileName::AbstractString = funcName * "RevAD")`

  - creates a MATLAB function `funcName` in a file `fileName.m` that carries out the reverse AD mode for the function recorded in `tape` by `record_tape`. This is difficult to do in MATLAB itself, and illustrates how to use a `CompGraph` in code generation.

### Example
The usage of `ReverseAD` is illustrated by the script [testRevAD.jl](test/testRevAD.jl), which can be run from the terminal if this repository is cloned. Consider again the following function, with a domain dimension of `2` and a range dimension of `1`.
```julia
f(x) = (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2
```

We may generate an AD tape for `f` by any of the following methoods, after `using .ReverseAD`. We may define `f` in advance and then load it into `record_tape`:
```julia
f(x) = (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2
tape = record_tape(f, 2, 1)
```
Or, we may load it in as a one-liner anonymous function:
```julia
tape = record_tape(x -> (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2, 2, 1)
```
Or, we may use Julia's `do` syntax, which might be handy for longer anonymous functions:
```julia
tape = record_tape(2, 1) do
  return (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2, 2, 1)
end
```
At this point, we may perform the reverse AD mode on `tape`. The following command computes the gradient `xBar` of `f` at `[2.0, 2.0]`:
```julia
_, xBar = reverse_AD!(tape, [2.0, 2.0], [1.0])
```
Alternatively, the following command generates a MATLAB file `fRevAD.m` that performs the reverse AD mode on `f`:
```julia
generate_revAD_matlab_code!(tape)
```
The file generated by this command is included in this repository as [fRevAD.m](test/fRevAD.m).

## References
to be written



