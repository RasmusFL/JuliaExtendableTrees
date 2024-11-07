# structs and functions to build trees

# global constants
#------------------------------------------------------------------------------------

const tree_types = ["Classification", "Regression", "Survival"]

# structs
#------------------------------------------------------------------------------------

struct ClassificationNode
    feature::Union{Int, Nothing}                      # index of the feature split upon
    threshold::Union{Float64, Nothing}                # the threshold for the split
    left::Union{ClassificationNode, Nothing}          # left daughter
    right::Union{ClassificationNode, Nothing}         # right daughter
    depth::Int                                        # depth of the node
    num::Int                                          # number of observations in the node
    val::Union{Int, Nothing}                          # predicted value of the node
end

struct RegressionNode
    feature::Union{Int, Nothing}                  # index of the feature split upon
    threshold::Union{Float64, Nothing}            # the threshold for the split
    left::Union{RegressionNode, Nothing}          # left daughter
    right::Union{RegressionNode, Nothing}         # right daughter
    depth::Int                                    # depth of the node
    num::Int                                      # number of observations in the node
    val::Union{Float64, Nothing}                  # predicted value of the node
end

struct SurvivalNode
    feature::Union{Int, Nothing}                # index of the feature split upon
    threshold::Union{Float64, Nothing}          # the threshold for the split
    left::Union{SurvivalNode, Nothing}          # left daughter
    right::Union{SurvivalNode, Nothing}         # right daughter
    depth::Int                                  # depth of the node
    num::Union{Int, Nothing}                    # number of observed events in the node
    val::Union{Matrix{Float64}, Nothing}        # predicted value of the node
end

struct Tree
    root::Union{ClassificationNode, RegressionNode, SurvivalNode}   # root node
    type::String                                                    # type (Classification, Regression, Survival)
    n_terminal_nodes::Int                                           # number of terminal nodes in the tree
end

# functions to compute the value in a terminal node
#------------------------------------------------------------------------------------

function value_Classification(y::Array{Int64})::Int64
    mode(y)
end

function value_Regression(y::Array{Float64})::Float64
    mean(y)
end

function value_Survival(y::Matrix{Float64})::Matrix{Float64}
    Nelson_Aalen(y)
end

# growing classification trees
#------------------------------------------------------------------------------------

# recursive function to build the tree with return value the root node and the number of terminal nodes
function tree_builder_Classification(X::Matrix{Float64}, y::Array{Int64}, depth::Int, L::Function, max_depth::Int, min_node_size::Int, n_features::Int, n_split::Int)

    #a node should be declared terminal if the max_depth is reached or if the min_node_size will be exceeded
    if depth == max_depth || size(X, 1) < 2*min_node_size
        return(ClassificationNode(nothing, nothing, nothing, nothing, depth, length(y), value_Classification(y)), 1)
    end
    
    left_ind, right_ind, feature, threshold = best_split_Classification(X, y, L, min_node_size, n_features, n_split)[1:4]
    
    # if the best split is no split, the node is also declared terminal
    if left_ind === nothing
        return(ClassificationNode(nothing, nothing, nothing, nothing, depth, length(y), value_Classification(y)), 1)
    end

    # make daughters
    left_node, n_terminal_nodes_left = tree_builder_Classification(X[left_ind, :], y[left_ind] , depth + 1, L, max_depth, min_node_size, n_features, n_split)
    right_node, n_terminal_nodes_right = tree_builder_Classification(X[right_ind, :], y[right_ind], depth + 1, L, max_depth, min_node_size, n_features, n_split)
    node = ClassificationNode(feature, threshold, left_node, right_node, depth, length(y), nothing)
    return(node, n_terminal_nodes_left + n_terminal_nodes_right)
end

# function for growing a Classification tree
function grow_tree_Classification(X::Matrix{Float64}, y::Array{Int64}, L::Function, max_depth::Int, min_node_size::Int, n_features::Int, n_split::Int)
    root, n_terminal_nodes = tree_builder_Classification(X, y, 1, L, max_depth, min_node_size, n_features, n_split)
    return(Tree(root, "Classification", n_terminal_nodes))
end

# growing regression trees
#------------------------------------------------------------------------------------

function tree_builder_Regression(X::Matrix{Float64}, y::Array{Float64}, depth::Int, L::Function, max_depth::Int, min_node_size::Int, n_features::Int, n_split::Int)
    #a node should be declared terminal if the max_depth is reached or if the min_node_size will be exceeded
    if depth == max_depth || size(X, 1) < 2*min_node_size
        return(RegressionNode(nothing, nothing, nothing, nothing, depth, length(y), value_Regression(y)), 1)
    end 

    left_ind, right_ind, feature, threshold = best_split_Regression(X, y, L, min_node_size, n_features, n_split)[1:4]

    # if the best split is no split, the node is also declared terminal
    if left_ind === nothing
        #println("Here, left_ind = nothing and right_ind = ", right_ind, ", feature = ", feature, ", depth = ", depth, ", threshold = ", threshold)
        return(RegressionNode(nothing, nothing, nothing, nothing, depth, length(y), value_Regression(y)), 1)
    end

    # make daughters
    left_node, n_terminal_nodes_left = tree_builder_Regression(X[left_ind, :], y[left_ind], depth + 1, L, max_depth, min_node_size, n_features, n_split)
    right_node, n_terminal_nodes_right = tree_builder_Regression(X[right_ind, :], y[right_ind], depth + 1, L, max_depth, min_node_size, n_features, n_split)
    return(RegressionNode(feature, threshold, left_node, right_node, depth, length(y), nothing), n_terminal_nodes_left + n_terminal_nodes_right)
end

function grow_tree_Regression(X::Matrix{Float64}, y::Array{Float64}, L::Function, max_depth::Int, min_node_size::Int, n_features::Int, n_split::Int)
    root, n_terminal_nodes = tree_builder_Regression(X, y, 1, L, max_depth, min_node_size, n_features, n_split)
    return(Tree(root, "Regression", n_terminal_nodes))
end

# growing survival trees
#------------------------------------------------------------------------------------

function tree_builder_Survival(X::Matrix{Float64}, y::Matrix{Float64}, depth::Int, L::Function, max_depth::Int, min_node_size::Int, n_features::Int, n_split::Int)
    # a node should be declared terminal if the max_depth is reached or if the number of true deaths falls below min_node_size
    if depth == max_depth || size(X, 1) < 2 * min_node_size
        return(SurvivalNode(nothing, nothing, nothing, nothing, depth, length(y), value_Survival(y)), 1)
    end

    left_ind, right_ind, feature, threshold, split_val = best_split_Survival(X, y, L, min_node_size, n_features, n_split)

    # if the best split is no split, the node is also declared terminal
    if split_val < 0
        return(SurvivalNode(nothing, nothing, nothing, nothing, depth, length(y), value_Survival(y)), 1)
    end
    
    # make daughters
    left_node, n_terminal_nodes_left = tree_builder_Survival(X[left_ind, :], y[left_ind, :], depth + 1, L, max_depth, min_node_size, n_features, n_split)
    right_node, n_terminal_nodes_right = tree_builder_Survival(X[right_ind, :], y[right_ind, :], depth + 1, L, max_depth, min_node_size, n_features, n_split)
    return(SurvivalNode(feature, threshold, left_node, right_node, depth, nothing, nothing), n_terminal_nodes_left + n_terminal_nodes_right)
end

function grow_tree_Survival(X::Matrix{Float64}, y::Matrix{Float64}, L::Function, max_depth::Int, min_node_size::Int, n_features::Int, n_split::Int)
    root, n_terminal_nodes = tree_builder_Survival(X, y, 1, L, max_depth, min_node_size, n_features, n_split)
    return(Tree(root, "Survival", n_terminal_nodes))
end

# additional tree-related helper functions
#------------------------------------------------------------------------------------

# helper function to print a single node as a string in one line
function print_node(node)::String
    depth::String = "Depth: " * string(node.depth)
    feature::String = " Feature: " * string(node.feature)
    threshold::String = " Threshold: " * string(node.threshold)
    leaf::String = ""
    val::String  = ""
    num::String = ""

    # if we have a terminal node
    if node.left === nothing
        feature = ""
        threshold = ""
        leaf = " (Leaf)"
        val = " Value: " * string(node.val)
        num = " #Observations: " * string(node.num)
    end
    
    return(depth * feature * threshold * val * num * leaf)

end

# recursive function to make a list of nodes as strings
function print_tree_helper(node, list::Array{String})
    push!(list, print_node(node))
    if node.left !== nothing               # not in a terminal node
        print_tree_helper(node.left, list)
        print_tree_helper(node.right, list)
    end
end
