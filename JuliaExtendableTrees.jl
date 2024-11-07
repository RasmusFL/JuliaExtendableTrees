# the only file needed to include when using JET
# contains all interface and wrapper functions

using Random, Base.Threads, StatsBase, CSV, DataFrames

include("criteria.jl")
include("tree.jl")
include("splitter.jl")
include("forest.jl")
include("predict.jl")
include("error.jl")

# interface functions for decision trees
#------------------------------------------------------------------------------------

function grow_tree(X, y, type::String, L::Function; max_depth = nothing, min_node_size = nothing, n_features = nothing, n_split = nothing)
    # set the hyperparameters to their default, depending on the type of tree
    if n_features === nothing
        n_features = Int(round(sqrt(size(X, 2))))
    end
    if n_split === nothing
        n_split = 0         # by default, we use every available split point
    end

    # make sure X is a Matrix{Float64}
    X = X * 1.0

    if type == "Classification"
        if max_depth === nothing
            max_depth = 10
        end
        if min_node_size === nothing
            min_node_size = 5
        end
        
        # convert y to a vector of integers (it is assumed that y has integers as elements)
        y = Int.(y)
        grow_tree_Classification(X, y, L, max_depth, min_node_size, n_features, n_split)

    elseif type == "Regression"
        if max_depth === nothing
            max_depth = 10
        end
        if min_node_size === nothing
            min_node_size = 10
        end

        # make sure y is a vector of Float64
        y = y * 1.0
        grow_tree_Regression(X, y, L, max_depth, min_node_size, n_features, n_split)

    elseif type == "Survival"
        if max_depth === nothing
            max_depth = 0
        end
        if min_node_size === nothing
            min_node_size = 15
        end

        # make sure y is a Matrix{Float64} (it is assumed that y is a matrix of numbers)
        y = y * 1.0
        grow_tree_Survival(X, y, L, max_depth, min_node_size, n_features, n_split)

    else
        throw("Error: Type of tree not recognised")
    end
end

# function to print the nodes of a tree to the terminal in more readable fashion
# NOTE: not very sophisticated, simply prints in arbitrary order
function print_tree(tree::Tree)::Array{String}
    list = String["Type of tree: " * tree.type, "List of nodes:"]
    print_tree_helper(tree.root, list)
    push!(list, "Number of nodes: " * string(length(list) - 2))
    push!(list, "Number of terminal nodes: " * string(tree.n_terminal_nodes))
    return(list)
end

# a simple wrapper function for growing forests of type Classification, Regression, Survival
function grow_forest(X, y, type::String, L::Function; max_depth = nothing, min_node_size = nothing, n_features = nothing, n_split = nothing, n_trees = nothing, sfrac = nothing, swr = nothing)
    # set the hyperparameters to their default, depending on the type of tree
    if n_features === nothing
        n_features = Int(round(sqrt(size(X, 2))))
    end
    if n_split === nothing
        n_split = 10
    end
    if sfrac === nothing
        sfrac = 0.7
    end
    if swr === nothing
        swr = false
    end
    
    # make sure X is a Matrix{Float64}
    X = X * 1.0

    if type == "Classification"
        if max_depth === nothing
            max_depth = 10
        end
        if min_node_size === nothing
            min_node_size = 5
        end
        if n_trees === nothing
            n_trees = 1000
        end
        
        # convert y to a vector of integers (it is assumed that y has integers as elements)
        y = Int.(y)
        grow_forest_Classification(X, y, L, max_depth, min_node_size, n_features, n_split, n_trees, sfrac, swr)

    elseif type == "Regression"
        if max_depth === nothing
            max_depth = 10
        end
        if min_node_size === nothing
            min_node_size = 10
        end
        if n_trees === nothing
            n_trees = 1000
        end

        # make sure y is a vector of Float64
        y = y * 1.0
        grow_forest_Regression(X, y, L, max_depth, min_node_size, n_features, n_split, n_trees, sfrac, swr)

    elseif type == "Survival"
        if max_depth === nothing
            max_depth = 0
        end
        if min_node_size === nothing
            min_node_size = 5
        end
        if n_trees === nothing
            n_trees = 500
        end

        # make sure y is a matrix of Float64
        y = y * 1.0
        grow_forest_Survival(X, y, L, max_depth, min_node_size, n_features, n_split, n_trees, sfrac, swr)

    else
        throw("Error: Type of tree not recognised")
    end
end

# wrapper function for computing all OOB predictions for a forest
# only works for forests of type "Classification", "Regression" or "Survival",
# so new types of forests need to be implemented manually
function OOB_predict(forest::Forest)
    if forest.type == "Classification"
        OOB_predict_Classification(forest)
    elseif forest.type == "Regression"
        OOB_predict_Regression(forest)
    elseif forest.type == "Survival"
        OOB_predict_Survival(forest)
    else
        throw("Error: Type of forest not recognised")
    end
end

# wrapper function for computing OOB error for a forest
# only works for forests of type "Classification", "Regression" or "Survival",
# so new types of forests need to be implemented manually
function OOB_error(forest::Forest)
    if forest.type == "Classification"
        OOB_error_Classification(forest)
    elseif forest.type == "Regression"
        OOB_error_Regression(forest)
    elseif forest.type == "Survival"
        OOB_error_Survival(forest)
    else
        throw("Error: Type of forest not recognised")
    end
end

# function for printing a forest to the terminal
function print_forest(forest::Forest)
    println("Type of forest: " * forest.type)
    println("Number of observations: ", size(X, 1))
    println("Number of features available: ", size(X, 2))
    println("Number of trees: ", length(forest.trees))
    if forest.max_depth != 0
        println("Maximum depth of a tree: ", forest.max_depth)
    else
        println("Maximum depth of a tree: No maximum depth")
    end

    println("Minimum size of a node: ", forest.min_node_size)
    
    if forest.n_split != 0
        println("Number of splitting points used: ", forest.n_split)
    else
       println("Number of splitting points used: All points") 
    end
    println("Number of features used in a split: ", forest.n_features)
    println("Average number of terminal nodes: ", forest.avr_number_terminal_nodes)
end
