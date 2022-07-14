#=
module CompGraphs
=================
The CompGraph type in this module is intended to hold the computational graph of a 
single composite function with vector-valued inputs and outputs. This graph 
expresses the composite function as a recipe of elemental operations. Computational 
graphs are required for implementing certain numerical methods, such as: 

- the standard reverse/adjoint mode of automatic differentiation (AD)

- the "branch-locking" method for efficient reverse-AD-like generalized
  differentiation by Khan (2018)
  https://doi.org/10.1080/10556788.2017.1341506

- the "cone-squashing" method for generalized differentiation 
  by Khan and Barton (2012, 2013)
  https://doi.org/10.1145/2491491.2491493

- the "reverse McCormick" convex relaxations incorporating constraint propagation, 
  by Wechsung et al. (2015)
  https://doi.org/10.1007/s10898-015-0303-6

"load_function!" in this module uses operator overloading to construct the 
computational graph of a finite composition of supported operations. Each 
node and the graph overall can also hold additional user-specified data, 
intended for use in methods like the reverse AD mode.

Written by Kamil Khan on February 10, 2022
=#
module CompGraphs

using Printf

export CompGraph, GraphNode

export load_function!, is_function_loaded

# structs; T and P are for smuggling in any sort of application-dependent data,
# but could easily be Any or Nothing for simplicity.
struct GraphNode{P}
    operation::Symbol
    parentIndices::Vector{Int}       # identify operands of "operation"
    constValue::Union{Float64, Int}  # only used when "operation" = :const or :^
    data::P                          # hold extra node-specific data
end

struct CompGraph{T, P}
    nodeList::Vector{GraphNode{P}}
    domainDim::Int              # domain dimension of graphed function
    rangeDim::Int               # range dimension of graphed function
    data::T                     # hold extra data not specific to any one node
end

struct GraphBuilder{P}
    index::Int
    graph::CompGraph
    changeMaxToAbs::Bool
end

# constructors
GraphNode{P}(op::Symbol, i::Vector{Int}, p::P) where P = GraphNode{P}(op, i, 0.0, p)

# the following constructor requires a constructor T() with no arguments
function CompGraph{T, P}(n::Int, m::Int) where {T, P}
    return CompGraph{T, P}(GraphNode{P}[], n, m, T())
end

# A GraphNode.operation can be any Symbol from the following lists
const unaryOpList = [:-, :inv, :exp, :log, :sin, :cos, :abs]
const standardBinaryOpList = [:+, :-, :*, :/, :^, :hypot]
const customBinaryOpList = [:max, :min]
const customOpList = [:input, :output, :const]   # input, output, and Float64 constant

# print graph or individual nodes
const opStringDict = Dict(
    :- => "neg",
    :inv => "inv",
    :exp => "exp",
    :log => "log",
    :sin => "sin",
    :cos => "cos",
    :abs => "abs",
    :+ => " + ",
    :- => " - ",
    :* => " * ",
    :/ => " / ",
    :^ => " ^ ",
    :max => "max",
    :min => "min",
    :hypot => "hyp",
    :input => "inp",
    :output => "out",
    :const => "con",
)

function Base.show(io::IO, node::GraphNode)
    parents = node.parentIndices
    nParents = length(parents)

    if (node.operation == :^) && (length(parents) == 1)
        opString = @sprintf " ^%1d" node.constValue
    else
        opString = opStringDict[node.operation]
    end

    if nParents <= 2
        oneParent(i::Int) = (nParents < i) ? "   " : @sprintf "%-3d" parents[i]
        parentString = oneParent(1) * "  " * oneParent(2)
    else
        parentString = string(parents)
    end

    if node.operation == :const
        dataString = @sprintf "const: % .3e" node.constValue
    else
        dataString = string(node.data)
    end
    
    return print(io, opString, " | ", parentString, " | ", dataString)
end


function Base.show(io::IO, graph::CompGraph)
    return begin
        println(" index | op  | parents  | data")
        println(" ------------------------------")
        for (i, node) in enumerate(graph.nodeList)
            @printf "   %3d | " i
            println(node)
        end
    end
end

# load in a function using operator overloading, and store its computational graph
function load_function!(
    f::Function,
    graph::CompGraph{T, P},
    initP::P;
    shouldMaxBeChangedToAbs::Bool = false
) where {T, P}
    
    empty!(graph.nodeList)
    
    # push new nodes for function inputs
    xGB = [GraphBuilder{P}(i, graph, shouldMaxBeChangedToAbs) for i=1:(graph.domainDim)]
    for xComp in xGB
        inputData = deepcopy(initP)
        inputNode = GraphNode{P}(:input, Int[], 0.0, inputData)
        push!(graph.nodeList, inputNode)
    end

    # push new nodes for all intermediate operations, using operator overloading
    yGB = f(xGB)
    if !(yGB isa Vector)
        yGB = [yGB]
    end

    # push new nodes for function outputs
    for yComp in yGB
        outputData = deepcopy(initP)
        outputNode = GraphNode{P}(:output, [yComp.index], 0.0, outputData)
        push!(graph.nodeList, outputNode)
    end
end

is_function_loaded(graph::CompGraph) = !isempty(graph.nodeList)

## let GraphBuilder construct a nodeList by operator overloading

# overload all unary operations in unaryOpList
for op in unaryOpList
    @eval begin
        function Base.$op(u::GraphBuilder{P}) where P
            parentGraph = u.graph
            prevNodes = parentGraph.nodeList
            
            newNodeData = deepcopy(prevNodes[u.index].data)
            newNode = GraphNode{P}(Symbol($op), [u.index], 0.0, newNodeData)
            push!(prevNodes, newNode)
            
            return GraphBuilder{P}(length(prevNodes), parentGraph, u.changeMaxToAbs)
        end
    end
end 

# overload all binary operations in binaryOpList.
# Uses nontrivial "promote" rules to push Float64 constants to the parent graph
function Base.promote(uA::GraphBuilder{P}, uB::Float64) where P
    parentGraph = uA.graph
    prevNodes = parentGraph.nodeList

    newNodeData = deepcopy(prevNodes[uA.index].data)
    newNode = GraphNode{P}(:const, Int[], uB, newNodeData)
    push!(prevNodes, newNode)

    return (uA, GraphBuilder{P}(length(prevNodes), parentGraph, uA.changeMaxToAbs))
end
function Base.promote(uA::Float64, uB::GraphBuilder{P}) where P
    return reverse(promote(uB, uA))
end
Base.promote(uA::GraphBuilder{P}, uB::GraphBuilder{P}) where P = (uA, uB)

for op in standardBinaryOpList
    @eval begin
        function Base.$op(uA::GraphBuilder{P}, uB::GraphBuilder{P}) where P
            parentGraph = uA.graph
            prevNodes = parentGraph.nodeList
            
            newNodeData = deepcopy(prevNodes[uA.index].data)
            newNode = GraphNode{P}(Symbol($op), [uA.index, uB.index], 0.0, newNodeData)
            push!(prevNodes, newNode)
            
            return GraphBuilder{P}(length(prevNodes), parentGraph, uA.changeMaxToAbs)
        end
    end
end

# max/min can be processed like other binary operands,
# or can be converted to abs using the identities:
#   max(x,y)==0.5*(x+y+abs(x-y));   min(x,y)==0.5*(x+y-abs(x-y))
function Base.max(uA::GraphBuilder{P}, uB::GraphBuilder{P}) where P
    if uA.changeMaxToAbs
        return 0.5*(uA + uB + abs(uA - uB))
    else
        parentGraph = uA.graph
        prevNodes = parentGraph.nodeList
        
        newNodeData = deepcopy(prevNodes[uA.index].data)
        newNode = GraphNode{P}(:max, [uA.index, uB.index], 0.0, newNodeData)
        push!(prevNodes, newNode)
        
        return GraphBuilder{P}(length(prevNodes), parentGraph, uA.changeMaxToAbs)
    end
end

function Base.min(uA::GraphBuilder{P}, uB::GraphBuilder{P}) where P
    if uA.changeMaxToAbs
        return 0.5*(uA + uB - abs(uA - uB))
    else
        parentGraph = uA.graph
        prevNodes = parentGraph.nodeList
        
        newNodeData = deepcopy(prevNodes[uA.index].data)
        newNode = GraphNode{P}(:min, [uA.index, uB.index], 0.0, newNodeData)
        push!(prevNodes, newNode)
        
        return GraphBuilder{P}(length(prevNodes), parentGraph, uA.changeMaxToAbs)
    end
end

for op in [standardBinaryOpList; customBinaryOpList]
    @eval begin
        function Base.$op(uA::GraphBuilder{P}, uB::Float64) where P
            return $op(promote(uA, uB)...)
        end

        function Base.$op(uA::Float64, uB::GraphBuilder{P}) where P
            return $op(promote(uA, uB)...)
        end
    end
end

# constant integer exponents are supported
function Base.:^(uA::GraphBuilder{P}, uB::Int) where P
    parentGraph = uA.graph
    prevNodes = parentGraph.nodeList
    
    newNodeData = deepcopy(prevNodes[uA.index].data)
    newNode = GraphNode{P}(:^, [uA.index], uB, newNodeData)
    push!(prevNodes, newNode)
    
    return GraphBuilder{P}(length(prevNodes), parentGraph, uA.changeMaxToAbs)
end

end # module
