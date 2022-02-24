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

using .CompGraphs, Printf

export record_tape, reverse_AD!, generate_revAD_matlab_code!

# struct for holding node-specific information in computational graph
mutable struct NodeData
    val::Float64           # value, computed during forward sweep
    bar::Float64           # adjoint, computed during reverse sweep
end

# default value of NodeData
NodeData() = NodeData(0.0, 0.0)

# called when printing NodeData
Base.show(io::IO, n::NodeData) = @printf(io, "val: % .3e,   bar: % .3e", n.val, n.bar)

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

# create a CompGraph "tape" of a provided funcntion for reverse AD
function record_tape(
    f::Function,
    domainDim::Int,
    rangeDim::Int
)
    tape = CompGraph{TapeData, NodeData}(domainDim, rangeDim)
    load_function!(f, tape, NodeData())
    tape.data.areBarsZero = false
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
    if op == :input
        v.val = t.x[t.iX]
        t.iX += 1
        
    elseif op == :output
        v.val = u(1).val
        t.y[t.iY] = v.val
        t.iY += 1
        
    elseif op == :const
        v.val = node.constValue

    elseif (op == :^) && (nParents == 1)
        # in this case the power is stored as node.constValue
        v.val = (u(1).val)^(node.constValue)
        
    elseif nParents == 1
        # handle all other unary operations
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
    if op == :input
        t.xBar[t.iX] = v.bar
        t.iX -= 1
        
    elseif op == :output
        v.bar = t.yBar[t.iY]
        t.iY -= 1
        u(1).bar += v.bar
        
    elseif op == :const
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

        elseif op == :^
            u(1).bar += v.bar * node.constValue * (u(1).val)^(node.constValue - 1)
            
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

        elseif op == :^
            u2Node = tape.nodeList[node.parentIndices[2]]
            if u2Node.operation == :const
                u(1).bar += v.bar * u2Node.constValue * (u(1).val)^(u2Node.constValue - 1)
            else
                throw(DomainError("x^y term with varying y; write this as exp(y*log(x)) instead"))
            end
        else
            throw(DomainError("unsupported elemental operation: " * String(op)))
        end
    else
        throw(DomainError("unsupported elemental operation: " * String(op)))
    end
end

# generate MATLAB code for performing the reverse AD mode on the graphed function
function generate_revAD_matlab_code!(
    tape::CompGraph{TapeData, NodeData},
    funcName::AbstractString = "f",
    fileName::AbstractString = funcName * "RevAD"
)
    # initial check
    (is_function_loaded(tape)) ||
        throw(DomainError("tape: hasn't been loaded with a function"))

    # generate MATLAB script
    open(fileName * ".m", "w") do file
        print(file, "% Computes y = " * funcName * "(x)")
        println(file, " and xBar = (D" * funcName * "(x))'*yBar, using the reverse mode of")
        println(file, """
                % automatic differentiation (AD). 
                %
                % x, y, xBar, and yBar are all column vectors of appropriate dimension.""")
        if tape.rangeDim == 1
            println(file, "% If yBar = [1.0], then xBar will be the gradient vector of "
                    * funcName * " at x.")
        end
        println(file, """
                %
                % This code was automatically generated by ReverseAD.jl.""")
        println(file, "function [y, xBar] = " * fileName * "(x, yBar)")
        println(file, "% initialize")
        println(file, "l = " * string(length(tape.nodeList)) * ";  % tape length")
        println(file, """
                v = zeros(l, 1);  % values at each node of computational graph
                vBar = zeros(size(v));  % adjoints at each node of computational graph""")
        println(file, "y = zeros(size(yBar));  % will be " * funcName * "(x)")
        println(file, "xBar = zeros(size(x));  % will be (D" * funcName * "(x))'*yBar")
        println(file, """
                
                % evaluate y with forward sweep through computational graph""")
        tape.data.iX = 1
        tape.data.iY = 1
        for (i, node) in enumerate(tape.nodeList)
            fwd_matlab_codeGen_step!(file, tape, i, node)
        end
        println(file, """
                
                % evaluate xBar with reverse sweep through computational graph""")
        for (i, node) in Iterators.reverse(enumerate(tape.nodeList))
            rev_matlab_codeGen_step!(file, tape, i, node)
        end
        println(file, """
                
                return""")
    end
end

function fwd_matlab_codeGen_step!(
    io::IO,
    tape::CompGraph{TapeData, NodeData},
    i::Int,
    node::GraphNode{NodeData}
)
    # convenience labels
    op = node.operation
    nParents = length(node.parentIndices)
    vStr = "v(" * string(i) * ")"
    uStr(j) = "v(" * string(node.parentIndices[j]) * ")"
    cStr = string(node.constValue)
    t = tape.data

    # compute node value based on operation type
    if op == :input
        println(io, vStr * " = x(" * string(t.iX) * ");")
        t.iX += 1
        
    elseif op == :output
        println(io, vStr * " = " * uStr(1) * ";")
        println(io, "y(" * string(t.iY) * ") = " * vStr * ";")
        t.iY += 1
        
    elseif op == :const
        println(io, vStr * " = " * cStr * ";")
        
    elseif nParents == 1
        if op == :-
            println(io, vStr * " = -" * uStr(1) * ";")

        elseif op == :inv
            println(io, vStr * " = inv(" * uStr(1) * ");")
            
        elseif op == :exp
            println(io, vStr * " = exp(" * uStr(1) * ");")
            
        elseif op == :log
            println(io, vStr * " = log(" * uStr(1) * ");")

        elseif op == :sin
            println(io, vStr * " = sin(" * uStr(1) * ");")

        elseif op == :cos
            println(io, vStr * " = cos(" * uStr(1) * ");")

        elseif op == :^
            # in this case the power is stored as node.constValue
            println(io, vStr * " = " * uStr(1) * ".^" * cStr * ";")
            
        else
            throw(DomainError("unsupported elemental operation: " * String(op)))
        end
    elseif nParents == 2
        if op == :+
            println(io, vStr * " = " * uStr(1) * " + " * uStr(2) * ";")
            
        elseif op == :-
            println(io, vStr * " = " * uStr(1) * " - " * uStr(2) * ";")
            
        elseif op == :*
            println(io, vStr * " = " * uStr(1) * " * " * uStr(2) * ";")

        elseif op == :/
            println(io, vStr * " = " * uStr(1) * " / " * uStr(2) * ";")

        elseif op == :^
            u2Node = tape.nodeList[node.parentIndices[2]]
            if u2Node.operation == :const
                println(io, vStr * " = " * uStr(1) * ".^" * cStr * ";")
            else
                throw(DomainError("x^y term with varying y; write this as exp(y*log(x)) instead"))
            end
        else
            throw(DomainError("unsupported elemental operation: " * String(op)))
        end
    else
        throw(DomainError("unsupported elemental operation: " * String(op)))
    end
end

function rev_matlab_codeGen_step!(
    io::IO,
    tape::CompGraph{TapeData, NodeData},
    i::Int,
    node::GraphNode{NodeData}
)
    # convenience labels
    op = node.operation
    nParents = length(node.parentIndices)
    vBarStr = "vBar(" * string(i) * ")"
    uStr(j) = "v(" * string(node.parentIndices[j]) * ")"
    uBarStr(j) = "vBar(" * string(node.parentIndices[j]) * ")"
    uEqUBarStr(j) = uBarStr(j) * " = " * uBarStr(j)
    uEqUPlusVBarStr(j) = uEqUBarStr(j) * " + " * vBarStr
    uEqUMinusVBarStr(j) = uEqUBarStr(j) * " - " * vBarStr
    cStr = string(node.constValue)
    cM1Str = string(node.constValue - 1)
    t = tape.data

    # compute node value based on operation type
    if op == :input
        t.iX -= 1
        println(io, "xBar(" * string(t.iX) * ") = " * vBarStr * ";")
        
    elseif op == :output
        t.iY -= 1
        println(io, vBarStr * " = yBar(" * string(t.iY) * ");")
        println(io, uEqUPlusVBarStr(1) * ";")
        
    elseif op == :const
        # do nothing in this case

    elseif nParents == 1
        if op == :-
            println(io, uEqUMinusVBarStr(1) * ";")

        elseif op == :inv
            println(io, uEqUMinusVBarStr(1) * "/(" * uStr(1) * "^2);")
            
        elseif op == :exp
            println(io, uEqUPlusVBarStr(1) * " * exp(" * uStr(1) * ");")
            
        elseif op == :log
            println(io, uEqUPlusVBarStr(1) * " / " * uStr(1) * ";")

        elseif op == :sin
            println(io, uEqUPlusVBarStr(1) * " * cos(" * uStr(1) * ");")

        elseif op == :cos
            println(io, uEqUMinusVBarStr(1) * " * sin(" * uStr(1) * ");")

        elseif op == :^
            println(io, uEqUPlusVBarStr(1)
                * " * " * cStr * " * " * uStr(1) * ".^(" * cM1Str * ");")
            
        else
            throw(DomainError("unsupported elemental operation: " * String(op)))
        end
    elseif nParents == 2
        if op == :+
            println(io, uEqUPlusVBarStr(1) * ";")
            println(io, uEqUPlusVBarStr(2) * ";")
            
        elseif op == :-
            println(io, uEqUPlusVBarStr(1) * ";")
            println(io, uEqUMinusVBarStr(2) * ";")
            
        elseif op == :*
            println(io, uEqUPlusVBarStr(1) * " * " * uStr(2) * ";")
            println(io, uEqUPlusVBarStr(2) * " * " * uStr(1) * ";")

        elseif op == :/
            println(io, uEqUPlusVBarStr(1) * "/" * uStr(2) * ";")
            println(io, uEqUMinusVBarStr(2) * " * " * uStr(1)
                    * " / (" * uStr(2) *  "^2);")

        elseif op == :^
            u2Node = tape.nodeList[node.parentIndices[2]]
            if u2Node.operation == :const
                powStr = string(u2Node.constValue)
                powM1Str = string(u2Node.constValue - 1)
                println(io, uEqUPlusVBarStr(1)
                        * " * " * powStr * " * (" * uStr(1) * ")^(" * powM1Str * ");")
            else
                throw(DomainError("x^y term with varying y; write this as exp(y*log(x)) instead"))
            end
        else
            throw(DomainError("unsupported elemental operation: " * String(op)))
        end
    else
        throw(DomainError("unsupported elemental operation: " * String(op)))
    end
end
    
end # module
