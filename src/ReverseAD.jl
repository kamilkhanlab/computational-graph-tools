#=
module ReverseAD
================
A quick implementation of the reverse mode of automatic differentiation (AD), which 
traverses a computational graph constructed by CompGraphs.jl. The reverse AD mode first 
evaluates a function by stepping forward through the graph, and then evaluates adjoint 
derivatives by stepping backward through the graph. 

Roughly follows the method description in Chapter 6 of "Evaluating Derivatives (2nd ed.)" 
by Griewank and Walther (2008).

Requires CompGraphs.jl in the same folder.

Written by Kamil Khan on February 12, 2022.
=#

module ReverseAD

include("CompGraphs.jl")

using .CompGraphs

export generate_tape, reverse_AD!

# struct for holding node-specific information
mutable struct NodeData
    val::Float64           # value, computed during forward sweep
    bar::Float64           # adjoint, computed during reverse sweep
end

# default value of NodeData
NodeData() = NodeData(0.0, 0.0)

# struct for holding information not specific to an individual node
mutable struct TapeData
    x::Vector{Float64}     # input value to graphed function
    y::Vector{Float64}     # output value of graphed function
    xBar::Vector{Float64}  # output of reverse AD mode
    yBar::Vector{Float64}  # input to reverse AD mode
    iX::Int                # next input component to be processed
    iY::Int                # next output component to be processed
    areBarsZero::Bool      # used to check if reverse AD mode is initialized
end

# default value of TapeData
TapeData() = TapeData(
    Float64[],             # x
    Float64[],             # y
    Float64[],             # xBar
    Float64[],             # yBar
    1,                     # iX
    1,                     # iY
    false                  # areBarsZero
)

# create a CompTape "tape" for reverse AD, and load in a function
function generate_tape(
    f::Function,
    domainDim::Int,
    rangeDim::Int
)
    tape = CompGraph{TapeData, NodeData}(domainDim, rangeDim)
    load_function!(tape, f, NodeData())
    return tape
end

# carry out reverse AD mode, using a forward evaluation sweep
# then a reverse adjoint sweep through a function's computational tape
function reverse_AD!(
    tape::CompGraph{TapeData, NodeData},
    x::Vector{Float64},
    yBar::Vector{Float64}
)
    # convenience label
    t = tape.data

    # reverse AD mode
    fwd_evaluation_sweep!(tape, x)
    rev_adjoint_sweep!(tape, yBar)
    
    return t.y, t.xBar
end

# forward sweep through tape, to compute each node.data.val
# and initialize each node.data.bar
function fwd_evaluation_sweep!(
    tape::CompGraph{TapeData, NodeData},
    x::Vector{Float64}
)
    # convenience label
    t = tape.data
    
    # initial checks
    (is_function_loaded(tape)) ||
        throw(DomainError("tape: hasn't been loaded with a function"))
    (length(x) == tape.domainDim) ||
        throw(DomainError("x: # components doesn't match tape's domainDim"))
    
    # initialize
    t.x = x
    t.y = zeros(tape.rangeDim)
    t.iX = 1
    t.iY = 1

    # update nodes and tape.data.y via a forward sweep through tape
    for node in tape.nodeList
        fwd_evaluation_step!(tape, node)
    end
    t.areBarsZero = true
end

function fwd_evaluation_step!(
    tape::CompGraph{TapeData, NodeData},
    node::GraphNode{NodeData}
)
    # convenience labels
    op = node.operation
    nParents = length(node.parentIndices)
    u(j) = tape.nodeList[node.parentIndices[j]].data
    v = node.data
    t = tape.data

    # compute node value based on operation type
    if op == :inp
        v.val = t.x[t.iX]
        t.iX += 1
        
    elseif op == :out
        v.val = u(1).val
        t.y[t.iY] = v.val
        t.iY += 1
        
    elseif op == :con
        v.val = node.constValue
        
    elseif nParents == 1
        # handle all unary operations
        v.val = eval(op)(u(1).val)
        
    elseif nParents == 2
        # handle all binary operations
        v.val = eval(op)(u(1).val, u(2).val)
        
    else
        throw(DomainError("unsupported elemental operation: " * String(op)))
    end

    # initialize adjoint
    v.bar = 0.0
end

# reverse sweep through tape, to evaluate each node.data.bar
function rev_adjoint_sweep!(
    tape::CompGraph{TapeData, NodeData},
    yBar::Vector{Float64}
)
    # convenience label
    t = tape.data
    
    # initial checks
    (is_function_loaded(tape)) ||
        throw(DomainError("tape: hasn't been loaded with a function"))
    (length(yBar) == tape.rangeDim) ||
        throw(DomainError("yBar: # components doesn't match tape's rangeDim"))
    
    # initialize
    t.xBar = zeros(tape.domainDim)
    t.yBar = yBar
    t.iX = tape.domainDim
    t.iY = tape.rangeDim
    if !(t.areBarsZero)
        for node in tape.nodeList
            node.data.bar = 0.0
        end
    end

    # update nodes and tape.data.xBar via a reverse sweep through tape
    for node in Iterators.reverse(tape.nodeList)
        rev_adjoint_step!(tape, node)
    end
    t.areBarsZero = false
end

function rev_adjoint_step!(
    tape::CompGraph{TapeData, NodeData},
    node::GraphNode{NodeData}
)
    # convenience labels
    op = node.operation
    nParents = length(node.parentIndices)
    u(j) = tape.nodeList[node.parentIndices[j]].data
    v = node.data
    t = tape.data

    # compute parent nodes' ".bars" based on operation type
    if op == :inp
        t.xBar[t.iX] = v.bar
        t.iX -= 1
        
    elseif op == :out
        v.bar = t.yBar[t.iY]
        t.iY -= 1
        u(1).bar += v.bar
        
    elseif op == :con
        # no parent nodes; do nothing in this case
        
    elseif nParents == 1
        if op == :-
            u(1).bar -= v.bar

        elseif op == :inv
            u(1).bar -= v.bar / ((u(1).val)^2)
            
        elseif op == :exp
            # use the fact that v.val == exp(u(1).val)
            u(1).bar += v.bar * v.val
            
        elseif op == :log
            u(1).bar += v.bar / u(1).val

        elseif op == :sin
            u(1).bar += v.bar * cos(u(1).val)

        elseif op == :cos
            u(1).bar -= v.bar * sin(u(1).val)
            
        else
            throw(DomainError("unsupported elemental operation: " * String(op)))
        end
    elseif nParents == 2
        if op == :+
            u(1).bar += v.bar
            u(2).bar += v.bar
            
        elseif op == :-
            u(1).bar += v.bar
            u(2).bar -= v.bar
            
        elseif op == :*
            u(1).bar += v.bar * u(2).val
            u(2).bar += v.bar * u(1).val

        elseif op == :/
            u(1).bar += v.bar / u(2).val
            u(2).bar -= v.bar * u(1).val / ((u(2).val)^2)
            
        else
            throw(DomainError("unsupported elemental operation: " * String(op)))
        end
    else
        throw(DomainError("unsupported elemental operation: " * String(op)))
    end
end
    
end # module
