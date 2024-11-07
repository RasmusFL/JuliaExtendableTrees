# structs and functions to grow forests

# structs
#------------------------------------------------------------------------------------

mutable struct Forest
    X::Matrix{Float64}                      # training data features
    y                                       # training data labels
    trees::AbstractArray{Tree}              # collection of trees
    type::String                            # type (Classification, Regression, Survival)
    max_depth::Int                          # maximum depth of a tree
    min_node_size::Int                      # maximum size of a node
    n_features::Int                         # number of features used in each split
    n_split::Int                            # maximum number of thresholds considered in each split
    bootstrap_indices::BitMatrix            # a matrix where each column is a bitvector indicating which datapoints are in the sample
    avr_number_terminal_nodes::Float64      # average number of terminal nodes in a tree
end

# helper functions
#------------------------------------------------------------------------------------

# a function for making the bootstrap datasets, n is the number of observations
function bootstrap_data(n::Int, n_trees::Int, sfrac::Float64, swr::Bool)
    bootstrap_indices = falses(n, n_trees)
    sample_indices = reduce(hcat, [sample(1:n, Int(round(n * sfrac)), replace = swr) for _ in 1:n_trees])
    for i in 1:n_trees
        bootstrap_indices[sample_indices[:, i], i] .= 1
    end
    
    # return the indices in both bitvector format and int format (the latter is
    # used to optimise performance)
    return bootstrap_indices, sample_indices
end
    
# growing classification forests
#------------------------------------------------------------------------------------

function grow_forest_Classification(X::Matrix{Float64}, y::Array{Int64}, L::Function, max_depth::Int = 10, min_node_size::Int = 5, n_features::Int = Int(round(sqrt(size(X, 2)))), n_split::Int = 10, n_trees::Int = 1000, sfrac::Float64 = 0.7, swr::Bool = false)
    # choosing n_features larger than the number of features simply means no randomness
    if n_features > size(X, 2)
        n_features = size(X, 2)
    end

    trees = Array{Tree}(undef, n_trees)
    n = size(X, 1)
    bootstrap_indices, sample_indices = bootstrap_data(n, n_trees, sfrac, swr)
    total_terminal_nodes = 0

    # grow the trees from the bootstrap data
    @threads for i in 1:n_trees
        trees[i] = grow_tree_Classification(X[sample_indices[:, i], :], y[sample_indices[:, i]], L, max_depth, min_node_size, n_features, n_split)
        total_terminal_nodes += trees[i].n_terminal_nodes
    end

    return(Forest(X, y, trees, "Classification", max_depth, min_node_size, n_features, n_split, bootstrap_indices, total_terminal_nodes/n_trees))
end

# growing regression forests
#------------------------------------------------------------------------------------

function grow_forest_Regression(X::Matrix{Float64}, y::Array{Float64}, L::Function, max_depth::Int = 10, min_node_size::Int = 10, n_features::Int = Int(round(sqrt(size(X, 2)))), n_split::Int = 10, n_trees::Int = 1000, sfrac::Float64 = 0.7, swr::Bool = false)
    # choosing n_features larger than the number of features simply means no randomness
    if n_features > size(X, 2)
        n_features = size(X, 2)
    end

    trees = Array{Tree}(undef, n_trees)
    n = size(X, 1)
    bootstrap_indices, sample_indices = bootstrap_data(n, n_trees, sfrac, swr)
    total_terminal_nodes = 0

    # grow the trees from the bootstrap data
    @threads for i in 1:n_trees
        trees[i] = grow_tree_Regression(X[sample_indices[:, i], :], y[sample_indices[:, i]], L, max_depth, min_node_size, n_features, n_split)
        total_terminal_nodes += trees[i].n_terminal_nodes
    end

    return(Forest(X, y, trees, "Regression", max_depth, min_node_size, n_features, n_split, bootstrap_indices, total_terminal_nodes/n_trees))
end

# growing survival forests
#------------------------------------------------------------------------------------

function grow_forest_Survival(X::Matrix{Float64}, y::Matrix{Float64}, L::Function, max_depth::Int = 0, min_node_size::Int = 10, n_features::Int = Int(round(sqrt(size(X, 2)))), n_split::Int = 10, n_trees::Int = 500, sfrac::Float64 = 0.7, swr::Bool = false)
    # choosing n_features larger than the number of features simply means no randomness
    if n_features > size(X, 2)
        n_features = size(X, 2)
    end

    trees = Array{Tree}(undef, n_trees)
    n = size(X, 1)
    bootstrap_indices, sample_indices = bootstrap_data(n, n_trees, sfrac, swr)
    total_terminal_nodes = 0

    # grow the trees from the bootstrap data
    @threads for i in 1:n_trees
        trees[i] = grow_tree_Survival(X[sample_indices[:, i], :], y[sample_indices[:, i], :], L, max_depth, min_node_size, n_features, n_split)
        total_terminal_nodes += trees[i].n_terminal_nodes
    end

    return(Forest(X, y, trees, "Survival", max_depth, min_node_size, n_features, n_split, bootstrap_indices, total_terminal_nodes/n_trees))
end
