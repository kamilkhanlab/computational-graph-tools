#=
module ConeSquashing
=====================
A quick implementation of:

Evaluating an Element of the Clarke Generalized Jacobian of a Piecewise
Differentiable Function, developed in the article:
KA Khan & PI Barton (2012),
https://doi.org/10.1145/2491491.2491493

This implementation uses automatic differentiation to numerically determine
the Jacobian of essentially active functions.

#cut this -----|
It traverses through a computational graph constructed by CompGraphs.jl
using the chain rule to calculate the directional derivative of each elemental
function that makes up the given function 'f'.

It then solves a system of linear equations to evaluate an element of the
generalized Jacobian of 'f' at some given vector of 'x'.
-------|

Requires CompGraphs.jl in the same folder.

Written by Maha Chaudhry on July 20th, 2022
Edited by Kamil Khan
=#

module ConeSquashing

include("CompGraphs.jl")

using .CompGraphs, Printf, LinearAlgebra

export tape_setup,
    eval_gen_derivative!

## assemble CompGraph "tape" and nodes

# struct for holding node-specific information in computational graph
mutable struct NodeData
    val::Float64             # value, computed during forward sweep
    dotVal::Vector{Float64}  # directional derivative value, computed during forward sweep
end #struct

# default value of NodeData
NodeData() = NodeData(0.0, [0.0])

# called when printing NodeData
Base.show(io::IO, n::NodeData) = @printf(io, "val: % .3e,   dotVal: %s", n.val, n.dotVal)

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

# default value of TapeData
TapeData() = TapeData(
    Float64[],                      # x
    Float64[],                      # y
    Array{Float64}(undef, 0, 0),    # qMatrix
    Array{Float64}(undef, 0, 0),    # yDot
    Array{Float64}(undef, 0, 0),    # yJac
    1,                              # iX
    1                               # iY
)

# create a CompGraph "tape" of a provided function
function tape_setup(
    f::Function,
    domainDim::Int64,     # number of inputs
    rangeDim::Int64       # number of outputs
)
    tape = CompGraph{TapeData, NodeData}(domainDim, rangeDim)
    load_function!(f, tape, NodeData(), shouldMaxBeChangedToAbs = true)

    return tape

end #function

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
    verbosity::Int64 = 0 # level of printed detail
)
    # convenience label
    t = tape.data

    # sweep evaluation through tape to calculate each node .val and .dotVal
    fwd_val_evaluation_sweep!(tape, x, sweepMode = MAIN_SWEEP, verbosity = verbosity)

    # solve system of equations yJac*qMatrix = yDot for yJac
    t.yJac = t.yDot/t.qMatrix

    return t.yJac, t.qMatrix
end #function

# calculates:
#   function evaluations for each elemental function of given function 'f'
#   directional derivatives for each elemental function of given function 'f'
function fwd_val_evaluation_sweep!(
    tape::CompGraph{TapeData, NodeData},
    x::Vector{Float64};
    sweepMode::SweepType = MAIN_SWEEP, # set sweep evaluation procedure
    nodeCount::Int64 = 0, # number of nodes requiring remedial evaluation for remedial sweep,
    verbosity::Int64 = 0  # level of printed detail
)
    # convenience label
    t = tape.data

    # initial checks
    (is_function_loaded(tape)) ||
        throw(DomainError("tape: hasn't been loaded with a function"))
    lenXIn = length(x)
    (lenXIn == tape.domainDim) ||
        throw(DomainError("x: # components doesn't match tape's domainDim"))

    # initialize
    t.iX = 1
    t.iY = 1

    if sweepMode == MAIN_SWEEP
        # initialize
        t.x = x
        t.y = zeros(tape.rangeDim)
        t.qMatrix = Matrix{Float64}(I(lenXIn))
        t.yDot = zeros(tape.rangeDim, tape.domainDim)

        # updates nodes, tape.data.y, and tape.data.yDot via forward sweep through tape
        for (i, nodeI) in enumerate(tape.nodeList)
            fwd_val_evaluation_step!(tape, nodeI, i, sweepMode = MAIN_SWEEP, verbosity = verbosity)
        end #for

    elseif sweepMode == REMEDIAL_SWEEP
        # re-evaluates nodes and tape.data.yDot upto nodeCount
        #    via forward sweep through tape given an adjusted qMatrix
        for (i, nodeI) in Iterators.take(enumerate(tape.nodeList), nodeCount)
            fwd_val_evaluation_step!(tape, nodeI, i; sweepMode = REMEDIAL_SWEEP)
        end #if
    end #if

    return t.y, t.yDot

end #function

function fwd_val_evaluation_step!(
    tape::CompGraph{TapeData, NodeData},
    node::GraphNode{NodeData},
    nodeCount::Int64; # tracks index of node being evaluated
    sweepMode::SweepType = MAIN_SWEEP,
    verbosity::Int64 = 0
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

        # abs is specific to the forward sweep.
        # keeping here avoids an endless remedial loop.
        elseif op == :abs
            v.val = abs(u(1).val)

            # initialize
            kStar, uStar = 0, 0.0
            trigger = false

            # check if qMatrix needs to be adjusted
            for (k, dotValK) in enumerate(u(1).dotVal)
                if kStar != 0 && uStar*dotValK < 0.0
                    if verbosity == 1
                        display(tape)
                        println("Remedial sweep triggered.\n")
                        println("Setting k* = ", kStar, " ; k = ", k,
                            " ; uj* = ", uStar, " ; uj = ", dotValK, "\n")
                    end #if

                    #adjust qMatrix
                    t.qMatrix[:, k] += abs(dotValK/uStar).*t.qMatrix[:, kStar]

                    # complete remedial sweep through tape for adjusted qMatrix
                    #   upto indicated node index
                    fwd_val_evaluation_sweep!(tape, t.x; sweepMode = REMEDIAL_SWEEP, nodeCount)

                    # shows remedial sweep was triggered;
                    #   avoids secondary trigger for the same k value
                    trigger = true
                    kStar, uStar = 0, 0.0
                end #if

                if dotValK != 0.0 && u(1).val == 0.0 && trigger == false
                    kStar, uStar = k, dotValK
                    # reset trigger for k = k + 1
                    trigger = false
                end #if
            end #for

            # calculate .dotVal for node when trigger = false
            v.dotVal = ((u(1).val >= 0.0) ? 1.0 : -1.0) * u(1).dotVal

        elseif nParents == 1
            # handle all other .val unary operations
            v.val = eval(op)(u(1).val)

        elseif nParents == 2
            # handle all .val binary operations
            v.val = eval(op)(u(1).val, u(2).val)

        else
            throw(DomainError(op, "unsupported elemental operation"))
        end #if

    elseif sweepMode == REMEDIAL_SWEEP && op == :abs
        # calculating intermediate abs evaluations
        v.dotVal = ((u(1).val >= 0.0) ? 1.0 : -1.0) * u(1).dotVal
    end #if

    # dotVals are calculated in both evaluation procedures
    if op == :input
        v.dotVal = t.qMatrix[t.iX, :]
        t.iX += 1

    elseif op == :output
        v.dotVal = u(1).dotVal
        t.yDot[t.iY, :] = v.dotVal
        t.iY += 1

    elseif op == :const
        v.dotVal = zeros(tape.domainDim)

    elseif (op == :^) && (nParents == 1)
        v.dotVal = node.constValue * (u(1).val)^(node.constValue-1) * u(1).dotVal

    elseif nParents == 1
        # handle all other .dotVal unary operations
        #   supported operations: [:-, :inv, :exp, :log, :sin, :cos]
        if op == :- #negative
            v.dotVal = -u(1).dotVal

        elseif op == :inv #reciprocal
            v.dotVal = -u(1).dotVal / ((u(1).val)^2)

        elseif op == :exp #exponential
            v.dotVal = exp(u(1).val) * u(1).dotVal

        elseif op == :log #logarithmic
            v.dotVal = u(1).dotVal / u(1).val

        elseif op == :sin #sine
            v.dotVal = cos(u(1).val) * u(1).dotVal

        elseif op == :cos #cosine
            v.dotVal = -sin(u(1).val) * u(1).dotVal
        end #if

    elseif nParents == 2
        # handle all .dotVal binary operations
        #   supported operations: [:+, :-, :*, :/, :^]
        if op in [:+, :-]
            v.dotVal = eval(op)(u(1).dotVal, u(2).dotVal)

        elseif op == :*
            v.dotVal = (u(2).val * u(1).dotVal) + (u(1).val * u(2).dotVal)

        elseif op == :/
            invVal = inv(u(2).val)
            invDotVal = (-u(2).dotVal / ((u(2).val)^2))
            v.dotVal = (invVal * u(1).dotVal) + (u(1).val * invDotVal)

        elseif op == :^
            throw(DomainError(op, "unsupported elemental operation x^y;",
                "rewrite as exp(y*log(x))"))
        end #if

    else
        throw(DomainError(op, "unsupported elemental operation"))
    end #if
end #function

end #module
