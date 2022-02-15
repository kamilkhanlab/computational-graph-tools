#=
module CompGraphs
=================
The CompGraph type in this module is intended to hold the computational graph of a single composite function with vector-valued inputs and outputs. This graph expresses the composite function as a recipe of elemental operations. Computational graphs are required for implementing certain numerical methods, such as: 

- the standard reverse/adjoint mode of automatic differentiation (AD)

- the "cone-squashing" method for generalized differentiation 
  by Khan and Barton (2012, 2013)
  (https://doi.org/10.1145/2491491.2491493)

- the "reverse McCormick" convex relaxations incorporating constraint propagation, 
  by Wechsung et al. (2015)
  (https://doi.org/10.1007/s10898-015-0303-6)

"load_function!" in this module uses operator overloading to construct the computational graph of a finite composition of supported operations. Each node and the graph overall can also hold additional user-specified data, intended for use in methods like the reverse AD mode.

Written by Kamil Khan on February 10, 2022
=#
module CompGraphs

export CompGraph, GraphNode

export load_function!, is_function_loaded

# structs; T and P are for smuggling in any sort of application-dependent data,
# but could easily be Any or Nothing for simplicity.
struct GraphNode{P}
    operation::Symbol
    parentIndices::Vector{Int}  # identify operands of "operation"
    constValue::Float64         # only used when "operation" = :con
    data::P                     # hold extra node-specific data
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
end

# constructors
GraphNode{P}(op::Symbol, i::Vector{Int}, p::P) where P = GraphNode{P}(op, i, 0.0, p)

# the following constructor requires a constructor T() with no arguments
function CompGraph{T, P}(n::Int, m::Int) where {T, P}
    return CompGraph{T, P}(GraphNode{P}[], n, m, T())
end


# A GraphNode.operation can be any Symbol from the following lists
unaryOpList = [:-, :inv, :exp, :log, :sin, :cos, :abs]
binaryOpList = [:+, :-, :*, :/, :max, :min, :hypot]
customOpList = [:inp, :out, :con]   # input, output, and Float64 constant

# print graph or individual nodes
opStringDict = Dict(
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
    :max => "max",
    :min => "min",
    :hypot => "hyp",
    :inp => "inp",
    :out => "out",
    :con => "con"
)

function Base.show(io::IO, node::GraphNode)
    return print(
        io,
        opStringDict[node.operation], " | ",
        isempty(node.parentIndices) ? "[none]" : node.parentIndices, " | ",
        (node.operation == :con) ? node.constValue : node.data
    )
end

function Base.show(io::IO, graph::CompGraph)
    return begin
        println(" index | op  | parents | data")
        println(" ------------------------------")
        for (i, node) in enumerate(graph.nodeList)
            print("   ", i, "   | ")
            println(node)
        end
    end
end

# load in a function using operator overloading, and store its computational graph
function load_function!(
    graph::CompGraph{T, P},
    f::Function,
    defaultP::P
) where {T, P}
    
    empty!(graph.nodeList)
    
    # push new nodes for function inputs
    xGB = GraphBuilder{P}[]
    for i = 1:(graph.domainDim)
        push!(xGB, GraphBuilder{P}(i, graph))
        push!(graph.nodeList, GraphNode{P}(:inp, Int[], deepcopy(defaultP)))
    end

    # push new nodes for all intermediate operations, using operator overloading
    yGB = f(xGB)

    # push new nodes for function outputs
    for yComp in yGB
        outputData = deepcopy(defaultP)
        outputNode = GraphNode{P}(:out, [yComp.index], outputData)
        push!(graph.nodeList, outputNode)
    end
end

is_function_loaded(graph::CompGraph) = !isempty(graph.nodeList)

## let GraphBuilder construct a nodeList by operator overloading

# overload unary operations in unaryOpList
macro define_GraphBuilder_unary_op_rule(op)
    opEval = eval(op)
    return quote
        function Base.$opEval(u::GraphBuilder{P}) where P
            parentGraph = u.graph
            prevNodes = parentGraph.nodeList
            
            newNodeData = deepcopy(prevNodes[u.index].data)
            newNode = GraphNode{P}($op, [u.index], newNodeData)
            push!(prevNodes, newNode)
            
            return GraphBuilder{P}(length(prevNodes), parentGraph)
        end
    end
end

for i in 1:length(unaryOpList)
    @eval @define_GraphBuilder_unary_op_rule $unaryOpList[$i]
end

# overload binary operations in binaryOpList.
# Uses nontrivial "promote" rules to push Float64 constants to the parent graph
function Base.promote(uA::GraphBuilder{P}, uB::Float64) where P
    parentGraph = uA.graph
    prevNodes = parentGraph.nodeList

    newNodeData = deepcopy(prevNodes[uA.index].data)
    newNode = GraphNode{P}(:con, Int[], uB, newNodeData)
    push!(prevNodes, newNode)

    return (uA, GraphBuilder{P}(length(prevNodes), parentGraph))
end
Base.promote(uA::Float64, uB::GraphBuilder{P}) where P = reverse(promote(uB, uA))
Base.promote(uA::GraphBuilder{P}, uB::GraphBuilder{P}) where P = (uA, uB)

macro define_GraphBuilder_binary_op_rule(op)
    opEval = eval(op)
    return quote
        function Base.$opEval(uA::GraphBuilder{P}, uB::GraphBuilder{P}) where P
            parentGraph = uA.graph
            prevNodes = parentGraph.nodeList
            
            newNodeData = deepcopy(prevNodes[uA.index].data)
            newNode = GraphNode{P}($op, [uA.index, uB.index], newNodeData)
            push!(prevNodes, newNode)
            
            return GraphBuilder{P}(length(prevNodes), parentGraph)
        end

        function Base.$opEval(uA::GraphBuilder{P}, uB::Float64) where P
            return $opEval(promote(uA, uB)...)
        end

        function Base.$opEval(uA::Float64, uB::GraphBuilder{P}) where P
            return $opEval(promote(uA, uB)...)
        end
    end
end

for i in 1:length(binaryOpList)
    @eval @define_GraphBuilder_binary_op_rule $binaryOpList[$i]
end

end # module