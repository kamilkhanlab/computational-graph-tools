#=
module ConeSquashing
=====================
A quick implementation of:

Evaluating an Element of the Clarke Generalized Jacobian of a Piecewise
Differentiable Function, developed in the article:
KA Khan & PI Barton (2012),
https://doi.org/10.1007/978-3-642-30023-3_11

This implementation uses automatic differentiation to numerically determine
the Jacobian of essentially active functions.

Requires CompGraphs.jl in the same folder.

Written by Maha Chaudhry on July 20th, 2022
Edited by Kamil Khan
=#

module ConeSquashing

include("CompGraphs.jl")

using .CompGraphs, Printf, LinearAlgebra

export record_tape,
    eval_gen_derivative!

## assemble CompGraph "tape" and nodes

# struct for holding node-specific information in computational graph
mutable struct NodeData
    val::Float64             # value, computed during forward sweep
    dot::Vector{Float64}     # directional derivative value, computed during forward sweep
end #struct

# called when printing NodeData
Base.show(io::IO, n::NodeData) = @printf(io, "val: % .3e,   dot: %s", n.val, n.dot)

# struct for holding information not specific to an individual node
mutable struct TapeData
    x::Vector{Float64}          # input value to graphed function
    y::Vector{Float64}          # output value to graphed function
    qMatrix::Matrix{Float64}    # input to forward sweep calculation of directional derivatives
    yDot::Matrix{Float64}       # output directional derivative value
    yJac::Matrix{Float64}       # output element of generalized Jacobian
    iX::Int64                   # next input component to be processed
    iY::Int64                   # next output component to be processed
end #struct

# create a CompGraph "tape" of a provided function
function record_tape(
    f::Function,
    domainDim::Int64,     # number of inputs
    rangeDim::Int64       # number of outputs
)
    # default value of TapeData
    tapeData = TapeData(
        zeros(domainDim),               # x
        zeros(rangeDim),                # y
        Matrix{Float64}(I(domainDim)),  # qMatrix
        zeros(rangeDim, domainDim),     # yDot
        zeros(1, domainDim),            # yJac
        1,                              # iX
        1                               # iY
    )

    # default value of NodeData
    nodeData = NodeData(0.0, zeros(domainDim))

    tape = CompGraph{TapeData, NodeData}(GraphNode{NodeData}[], domainDim,
        rangeDim, tapeData)
    load_function!(f, tape, nodeData, shouldMaxBeChangedToAbs = true)

    return tape

end #function

# default tolerances; feel free to change.
const TOLERANCE = 1e-8   # used to decide if we're at the kink of "abs" or "hypot"

## assemble evaluation procedures for CompGraph "tape"

# implemented CompGraph sweep evaluation procedures
@enum SweepType begin
    MAIN_SWEEP       # forward sweep
    REMEDIAL_SWEEP   # remedial sweep for :abs operations
end

# calculate output element of generalized Jacobian using provided CompGraph "tape"
function eval_gen_derivative!(
    tape::CompGraph{TapeData, NodeData},
    x::Vector{Float64};
    kwargs...
)
    # convenience label
    t = tape.data

    # sweep evaluation through tape to calculate each node .val and .dot
    forward_sweep!(tape, x; sweepMode = MAIN_SWEEP, kwargs...)

    # solve system of equations yJac*qMatrix = yDot for yJac
    t.yJac = t.yDot/t.qMatrix

    return t.yJac, t.qMatrix
end #function

# calculates:
#   function evaluations for each elemental function of given function 'f'
#   directional derivatives for each elemental function of given function 'f'
function forward_sweep!(
    tape::CompGraph{TapeData, NodeData},
    x::Vector{Float64};
    sweepMode::SweepType = MAIN_SWEEP, # set sweep evaluation procedure
    currentNodeI::Int64 = 0, # number of nodes requiring remedial evaluation for remedial sweep,
    kwargs...
)
    # convenience label
    t = tape.data

    # initial checks
    (is_function_loaded(tape)) ||
        throw(DomainError("tape: hasn't been loaded with a function"))
    (length(x) == tape.domainDim) ||
        throw(DomainError("x: # components doesn't match tape's domainDim"))

    # initialize
    t.iX = 1
    t.iY = 1

    if sweepMode == MAIN_SWEEP
        # initialize
        t.x = x
        
        # updates nodes, tape.data.y, and tape.data.yDot via forward sweep through tape
        for (i, nodeI) in enumerate(tape.nodeList)
            forward_step!(tape, nodeI, i; sweepMode = MAIN_SWEEP, kwargs...)
        end #for

    elseif sweepMode == REMEDIAL_SWEEP
        # re-evaluates nodes and tape.data.yDot upto currentNodeI
        #    via forward sweep through tape given an adjusted qMatrix
        for (i, nodeI) in Iterators.take(enumerate(tape.nodeList), currentNodeI)
            forward_step!(tape, nodeI, i; sweepMode = REMEDIAL_SWEEP)
        end #if
    end #if

    return t.y, t.yDot, t.qMatrix

end #function

function forward_step!(
    tape::CompGraph{TapeData, NodeData},
    node::GraphNode{NodeData},
    currentNodeI::Int64; # tracks index of node being evaluated
    sweepMode::SweepType = MAIN_SWEEP,
    verbosity::Int64 = 0 # level of printed detail
)
    # convenience labels
    op = node.operation
    nParents = length(node.parentIndices)
    u(j) = tape.nodeList[node.parentIndices[j]].data
    v = node.data
    t = tape.data

    if sweepMode == MAIN_SWEEP
        # compute node value based on operation type
        if op == :input
            v.val = v.val = t.x[t.iX]

        elseif op == :output
            v.val = u(1).val
            t.y[t.iY] = v.val

        elseif op == :const
            v.val = node.constValue

        elseif (op == :^) && (nParents == 1)
            # in this case the power is stored as node.constValue
            v.val = (u(1).val)^(node.constValue)

        elseif op == :abs
            v.val = abs(u(1).val)

            # initialize
            kStar = 0
            uStar = 0.0
            trigger = false

            # adjusts qMatrix and updates CompGraph as needed
            for (k, dotK) in enumerate(u(1).dot)
                if kStar != 0 && uStar*dotK < 0.0
                    if verbosity == 1
                        display(tape)
                        println("Remedial sweep triggered.\n")
                        println("Setting k* = ", kStar, " ; k = ", k,
                            " ; uj* = ", uStar, " ; uj = ", dotK, "\n")
                    end #if

                    #adjusts qMatrix
                    t.qMatrix[:, k] += abs(dotK/uStar).*t.qMatrix[:, kStar]

                    # complete remedial sweep through tape for adjusted qMatrix
                    #   upto indicated node index
                    forward_sweep!(tape, t.x; sweepMode = REMEDIAL_SWEEP, currentNodeI)
                end #if

                if !(-TOLERANCE <= dotK <= TOLERANCE) && (-TOLERANCE <= u(1).val <= TOLERANCE) && trigger == false

                    kStar = k
                    uStar = dotK
                    # shows kStar and uStar have been set
                        # avoids re-adjustment of uStar and kStar within this loop
                    trigger = true
                end #if
            end #for

        elseif nParents == 1
            # handle all other .val unary operations
            v.val = eval(op)(u(1).val)

        elseif nParents == 2
            # handle all .val binary operations
            v.val = eval(op)(u(1).val, u(2).val)

        else
            throw(DomainError(op, "unsupported elemental operation"))
        end #if
    end #if

    # dots are calculated in both evaluation procedures
    if op == :input
        v.dot = t.qMatrix[t.iX, :]
        t.iX += 1

    elseif op == :output
        v.dot = u(1).dot
        t.yDot[t.iY, :] = v.dot
        t.iY += 1

    elseif op == :const
        # handles constants
        v.dot = zeros(tape.domainDim)

    elseif nParents == 1
        # handle all unary .dot operations
        pd = partial_deriv(op, [u(1).val]; constValue = node.constValue)
        v.dot = pd * u(1).dot

    elseif nParents == 2
        # handle all binary .dot operations
        pd = partial_deriv(op, [u(1).val, u(2).val]; constValue = node.constValue)
        v.dot = pd[1] * u(1).dot + pd[2] * u(2).dot

    else
        throw(DomainError(op, "unsupported elemental operation"))
    end #if

end #function

function partial_deriv(
    op::Symbol,
    val::Vector{Float64};
    constValue::Union{Int64, Float64}
    )

    # initialize
    pd = 0.0

    if (op == :^) && (length(val) == 1)
        pd = constValue * (val[1])^(constValue-1)

    elseif length(val) == 1
        if op == :- #negative
            pd = -1.0

        elseif op == :inv #reciprocal
            pd = -1.0 / (val[1]^2)

        elseif op == :exp #exponential
            pd = exp(val[1])

        elseif op == :log #logarithmic
            pd = 1.0 / val[1]

        elseif op == :sin #sine
            pd = cos(val[1])

        elseif op == :cos #cosine
            pd = -sin(val[1])

        elseif op == :abs
            pd = ((val[1] >= 0.0) ? 1.0 : -1.0)

        elseif op == :^
            pd = constValue * (val[1])^(constValue-1)

        else
            throw(DomainError(op, "unsupported elemental operation"))
        end #if

    elseif length(val) == 2
        if op == :+ #addition
            pd = [1.0, 1.0]

        elseif op == :- #subtraction
            pd = [1.0, -1.0]

        elseif op == :* #multiplication
            pd = [val[2], val[1]]

        elseif op == :/ #division
            pd = [inv(val[2]), -val[1]/(val[2]^2)]

        elseif op == :^ #exponential
            throw(DomainError(op, "unsupported elemental operation x^y; rewrite as exp(y*log(x))"))
        end #if

    else
        throw(DomainError(op, "unsupported elemental operation"))
    end #if

    return pd

end #function

end #module
