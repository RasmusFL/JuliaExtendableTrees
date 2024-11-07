# old code

function Harrell_error(forest::Forest, X_test, y_test, predicted = nothing)
    if predicted === nothing
        predicted = predict(forest, X_test, true)
    end
    n = length(predicted)

    # compute the outcomes
    outcomes = Array{Float64}(undef, n)
    for i in 1:n
        outcomes[i] = sum(predicted[i][:, 2])   # fixed from [2] to [:, 2]
    end

    # form a matrix of all pairs
    pairs = [(s, t) for s in y_test[:, 1], t in y_test[:, 1]]
    p = size(pairs, 1)

    # compute the C-index
    Concordance = 0
    Permissible = 0

    for i in 1:p
        for j in 1:p
            # (T_i, T_j) with T_i < T_j is permissible if delta_i = 1
            if pairs[i, j][1] < pairs[i, j][2] && y[i, 2] == 1
                Permissible += 1
                if outcomes[i] == outcomes[j]
                    Concordance += 0.5
                elseif outcomes[i] > outcomes[j]    # the shorter survival times has worse predicted outcome
                    Concordance += 1
                end
                
            # same case as above but with the roles of T_i and T_j switched
            elseif pairs[i, j][1] > pairs[i, j][2] && y[j, 2] == 1
                Permissible += 1
                if outcomes[i] == outcomes[j]
                    Concordance += 0.5
                elseif outcomes[i] < outcomes[j]    # the shorter survival times has worse predicted outcome
                    Concordance += 1
                end
            
            # (T_i, T_j) with T_i = T_j is permissible if at least one is a death
            elseif pairs[i, j][1] == pairs[i, j][2] && y[i, 2] + y[j, 2] > 0
                Permissible += 1
                if y[i, 2] + y[1, 2] == 2           # both are deaths
                    if outcomes[i] == outcomes[j]
                        Concordance += 1
                    else
                        Concordance += 0.5
                    end
                elseif y[i, 2] == 1 && y[j, 2] == 0
                    if outcomes[i] > outcomes[j]
                        Concordance += 1
                    else
                        Concordance += 0.5
                    end
                else
                    if outcomes[i] < outcomes[j]
                        Concordance += 1
                    else
                        Concordance += 0.5
                    end
                end
            end
        end
    end
    # returns the error E = 1 - Concordance/Permissible
    1 - Concordance/Permissible
end

function predict_Survival_old(forest::Forest, x, OOB::Bool = false)
    predictions = predicted_labels(forest, x, OOB)
    
    t = sort(unique(forest.y[forest.y[:, 2] .== 1, :][:, 1]))
    N = length(t)
    B = length(predictions)

    # compute the values for each point in the ensemble domain
    values = zeros(Float64, N)
    for i in 1:B
        val = predictions[i]
        current = 1
        for j in 1:size(val, 1)
            for k in current:N
                if val[j, 1] == t[k]
                    values[k] += val[j, 2]
                    current = k + 1
                    continue
                end
            end
        end
    end

    t, values/B
end

# old code for growing trees
#------------------------------------------------------------------------------------

# indices indicates which rows of X should be included, e.g. [0, 1, 1, 0, 0, 1], while
# new_indices indicates the rows included after a split, e.g. [0, 1, 1]
# this function gives indices in terms of the original data set, e.g. [0, 0, 1, 0, 0, 1]
# for this example
function indices_split(indices::BitArray{1}, new_indices::BitArray{1})

    res = copy(indices)
    j = 0
    for i in 1:length(indices)
        if indices[i] == 1
            j = j + 1
            if new_indices[j] == 0
                res[i] = 0
            end
        end
    end

    res
end

# recursive function to build the tree with return value the root node and the number of terminal nodes
function tree_builder(X, y, indices, depth, type::String, L, max_depth::Int, min_node_size::Int, n_features::Int, n_split::Int)

    # in the case of Classification or Regression, a node should be declared terminal if the max_depth is reached
    # or if the min_node_size will be exceeded
    if type == "Classification" && (depth == max_depth || size(X[indices, :], 1) < 2*min_node_size)
        val = value_Classification(y[indices, :])
        node = Node(nothing, nothing, nothing, nothing, indices, depth, val)
        return(node, 1)
    elseif type == "Regression" && (depth == max_depth || size(X[indices, :], 1) < 2*min_node_size)
        val = value_Regression(y[indices, :])
        node = Node(nothing, nothing, nothing, nothing, indices, depth, val)
        return(node, 1)
    
    # in the case of Survival, a node should be declared terminal if the max_depth is reached or the number of
    # events (true deaths) will exceed min_node_size
    elseif type == "Survival" && (depth == max_depth || sum(y[indices, :][:, 2]) < 2*min_node_size)
        val = value_Survival(y[indices, :])
        node = Node(nothing, nothing, nothing, nothing, indices, depth, val)
        return(node, 1)
    end
    
    left_ind, right_ind, feature, threshold, split_val = best_split(X[indices, :], y[indices, :], type, L, min_node_size, n_features, n_split)
    
    if left_ind === nothing
        if type == "Classification"
            val = value_Classification(y[indices, :])
        elseif type == "Regression"
            val = value_Regression(y[indices, :])
        elseif type == "Survival"
            val = value_Survival(y[indices, :])
        else
            throw("Error: Type of tree not recognised")
        end
        
        node = Node(nothing, nothing, nothing, nothing, indices, depth, val)
        return(node, 1)
    end

    node = Node(feature, threshold, nothing, nothing, indices, depth, nothing)

    left_ind = indices_split(indices, left_ind)
    right_ind = indices_split(indices, right_ind)

    # make daughters
    node.left, n_terminal_nodes_left = tree_builder(X, y, left_ind, depth + 1, type, L, max_depth, min_node_size, n_features, n_split)
    node.right, n_terminal_nodes_right = tree_builder(X, y, right_ind, depth + 1, type, L, max_depth, min_node_size, n_features, n_split)
    return(node, n_terminal_nodes_left + n_terminal_nodes_right)
end

# grows a tree from data
# (X, y) is the data used to grow the tree
# L is the function used to evaluate the split, the higher L, the better split
# max_depth is the maximum depth of the tree (max_depth = 0 means no maximum depth)
# min_node_size is the minimum number of data points in a node (NOTE: for
# survival trees, it is the minimum number of true deaths in a terminal node)
# n_features is the number of features randomly selected in each split
# n_split is the number of threshold values randomly selected in each split
# n_split = 0 means that all values are used, if n_split is higher than the number of
# unique values, all values are used
function grow_tree(X, y, type::String, L, max_depth::Int, min_node_size::Int, n_features::Int, n_split::Int)
    if !in(type, tree_types)
        throw("Error: Type of tree not recognised")
    end
    root, n_terminal_nodes = tree_builder(X, y, fill(true, size(X, 1)), 1, type, L, max_depth, min_node_size, n_features, n_split)
    return(Tree(root, type, n_terminal_nodes))
end


# old code for splitting in the Classification setup
#------------------------------------------------------------------------------------

# finds the best split in the Classification regime
function best_split_Classification(X::Matrix{Float64}, y::Array{Int64}, L::Function, min_node_size::Int, n_features::Int, n_split::Int)
    best_feature = nothing
    best_threshold = nothing
    best_split_val = -Inf
    best_left_ind = nothing
    best_right_ind = nothing

    # randomly select n_features features to split on
    features = sample(1:size(X, 2), n_features, replace = false)
    for i in features
        # select the unique values for the given feature
        thresholds = unique(X[:, i])

        # only use n_split different threshold values (all are used if n_split = 0
        # or n_split is larger than the number of unique values)
        if n_split != 0 && n_split < length(thresholds)
            thresholds = sort(sample(thresholds, n_split, replace = false))
        end

        n = length(thresholds)
        mid = Int(ceil(n/2))

        # check the first half of the splits
        for j in 1:(mid - 1)

            # if the current value is equal to the previous one, skip this
            # iteration of the loop (works because thresholds is sorted)
            if j + 1 < mid && thresholds[mid - j] == thresholds[mid - (j + 1)]
                continue
            end

            left_ind, right_ind = split_dataset(X, i, thresholds[mid - j])

            # if the split creates a node with too few data points, skip the split
            # entirely and all following splits (depends on the type of tree)
            if sum(left_ind) < min_node_size || sum(right_ind) < min_node_size
                break
            end

            l = L(X, y, left_ind, right_ind)
            if l > best_split_val
                best_threshold = thresholds[mid - j]
                best_feature = i
                best_split_val = l
                best_left_ind = left_ind
                best_right_ind = right_ind
            end
        end

        # check the second half of the splits
        for j in mid:n

            # if the current value is equal to the previous one, skip this
            # iteration of the loop (works because thresholds is sorted)
            if j < n && thresholds[j] == thresholds[j + 1]
                continue
            end

            left_ind, right_ind = split_dataset(X, i, thresholds[j])

            # if the split creates a node with too few data points, skip the split
            # entirely and all following splits
            if sum(left_ind) < min_node_size || sum(right_ind) < min_node_size
                break
            end

            l = L(X, y, left_ind, right_ind)
            if l > best_split_val
                best_threshold = thresholds[j]
                best_feature = i
                best_split_val = l
                best_left_ind = left_ind
                best_right_ind = right_ind
            end
        end
        
    end

    # return the overall best split
    return best_left_ind, best_right_ind, best_feature, best_threshold, best_split_val
end

# grows a forest from the data (X, y) of type type, criteria function L to determine split values,
# hyperparameters: max_depth, min_node_size, n_features, n_split, n_trees, sfrac, swr
# applies multithreading by default, but you need to manually set the number of threads in Julia
function grow_forest_old(X, y, type::String, L, max_depth::Int = 5, min_node_size::Int = 2, n_features::Int = Int(round(log2(size(X, 2)))), n_split::Int = 5, n_trees::Int = 500, sfrac::Float64 = 0.7, swr::Bool = false)

    # choosing n_features larger than the number of features simply means no randomness
    if n_features > size(X, 2)
        n_features = size(X, 2)
    end

    trees = Array{Tree}(undef, n_trees)
    n = size(X, 1)
    bootstrap_indices = Array{BitArray}(undef, n_trees)
    total_terminal_nodes = 0
    @threads for i in 1:n_trees
        # bootstrap the data
        sample_indices = sample(1:n, Int(round(n * sfrac)), replace = swr)
        X_sample = X[sample_indices, :]
        y_sample = y[sample_indices, :]

        # save the bitvector indicating which observations from the original data are kept
        bootstrap_ind = falses(n)
        for j in sample_indices
            bootstrap_ind[j] = true
        end

        # grow the tree from the bootstrap data
        trees[i] = grow_tree(X_sample, y_sample, type, L, max_depth, min_node_size, n_features, n_split)
        bootstrap_indices[i] = bootstrap_ind
        total_terminal_nodes += trees[i].n_terminal_nodes
    end
    
    return(Forest(X, y, trees, type, max_depth, min_node_size, n_features, n_split, bootstrap_indices, total_terminal_nodes/n_trees))
end

# old code for computing the C-index
#------------------------------------------------------------------------------------

# general function for computing Harrell's C-index for a vector of outcomes and corresponding
# survival data y (this function may be wrong??)
function Harrell_C_old(outcomes::Vector{Float64}, y_test::Matrix{Float64})::Float64
    # initialise numerator and denominator
    Concordance = 0
    Permissible = 0
    n = length(outcomes)

    # sort observations and outcomes by time to event, where uncensored observations come first
    obs = hcat(copy(y_test), outcomes)
    #return(obs)
    obs[:, 2] = map(x -> 1 - x, obs[:, 2])
    obs = sortslices(obs, dims = 1, by = x -> (x[1], x[2]))

    for i in 1:n
        # if subject i is censored, all pairs (i, j) are not permissible
        if obs[i, 2] == 0
            continue
        end
        for j in (i + 1):n
            # if delta_i = 1, the pair (i, j) is permissible if T_i < T_j or
            # T_i = T_j j is censored
            if obs[i, 1] < obs[j, 1] || (obs[i, 1] == obs[j, 1] && obs[j, 2] == 0)
                Permissible += 1
                # if i has higher risk than j, the model guesses correctly
                if obs[i, 3] > obs[j, 3]
                    Concordance += 1
                # if the risk is the same, the pair is neither concordant or
                # discordant, so add 0.5 to the numerator
                elseif obs[i, 3] == obs[j, 3]
                    Concordance += 0.5
                end
            end
        end
    end

    #return(Concordance, Permissible, Concordance/Permissible)
    Concordance/Permissible
end

# old splitter code
#------------------------------------------------------------------------------------

# returns the best split and the best feature and threshold (OLD FUNCTION)
# (X, y) is the data in the current node
# L is the function used to evaluate the split, the higher L, the better split
# min_node_size is the minimum number of data points in a node (NOTE: for
# survival trees, it is the minimum number of true deaths in a terminal node)
# n_features is the number of features randomly selected in each split
# n_split is the number of threshold values randomly selected in each split
# n_split = 0 means that all values are used, if n_split is higher than the number of
# unique values, all values are used
function best_split(X, y, type::String, L, min_node_size::Int, n_features::Int, n_split::Int)

    best_feature = nothing
    best_threshold = nothing
    best_split_val = -Inf
    best_left_ind = nothing
    best_right_ind = nothing

    # randomly select n_features features to split on
    features = sample(1:size(X, 2), n_features, replace = false)
    for i in features
        # select the unique values for the given feature
        thresholds = unique(X[:, i])

        # only use n_split different threshold values (all are used if n_split = 0
        # or n_split is larger than the number of unique values)
        if n_split != 0 && n_split < length(thresholds)
            thresholds = sort(sample(thresholds, n_split, replace = false))
        end

        n = length(thresholds)
        mid = Int(ceil(n/2))

        # iterate through all unique values of the feature to find the best split

        # check the second half of the splits
        for j in mid:n

            # if the current value is equal to the previous one, skip this
            # iteration of the loop (works because thresholds is sorted)
            if j < n && thresholds[j] == thresholds[j + 1]
                continue
            end

            left_ind, right_ind = split_dataset(X, indices, i, thresholds[j])

            # if the split creates a node with too few data points, skip the split
            # entirely and all following splits (depends on the type of tree)
            if type == "Classification" || type == "Regression"
                if sum(left_ind) < min_node_size || sum(right_ind) < min_node_size
                    break
                end
            end
            if type == "Survival"
                if sum(y[left_ind, :][:, 2]) < min_node_size || sum(y[right_ind, :][:, 2]) < min_node_size
                    break
                end
            end

            l = L(X, y, i, thresholds[j])
            if l > best_split_val
                best_feature = i
                best_threshold = thresholds[j]
                best_split_val = l
                best_left_ind = left_ind
                best_right_ind = right_ind
            end
        end

        # check the first half of the splits
        for j in 1:(mid - 1)

            # if the current value is equal to the previous one, skip this
            # iteration of the loop (works because thresholds is sorted)
            if j + 1 < mid && thresholds[mid - j] == thresholds[mid - (j + 1)]
                continue
            end

            left_ind, right_ind = split_dataset(X, indices, i, thresholds[mid - j])

            # if the split creates a node with too few data points, skip the split
            # entirely and all following splits (depends on the type of tree)
            if type == "Classification" || type == "Regression"
                if sum(left_ind) < min_node_size || sum(right_ind) < min_node_size
                    break
                end
            end
            if type == "Survival"
                if sum(y[left_ind, :][:, 2]) < min_node_size || sum(y[right_ind, :][:, 2]) < min_node_size
                    break
                end
            end

            l = L(X, y, i, thresholds[mid - j])
            if l > best_split_val
                best_feature = i
                best_threshold = thresholds[mid - j]
                best_split_val = l
                best_left_ind = left_ind
                best_right_ind = right_ind
            end
        end
    end

    # return the best split
    return best_left_ind, best_right_ind, best_feature, best_threshold, best_split_val
end

function best_split_Survival_new(X::Matrix{Float64}, y::Matrix{Float64}, L::Function, min_node_size::Int, n_features::Int, n_split::Int)
    best_feature = nothing
    best_threshold = nothing
    best_split_val = -1
    best_left_ind = nothing
    best_right_ind = nothing

    features = sample(1:size(X, 2), n_features, replace = false)
    for i in features
        thresholds = unique(X[:, i])
        if n_split != 0 && n_split < length(thresholds)
            thresholds = sample(thresholds, n_split, replace = false)
        end
        n_i = length(thresholds)
        for j in 1:n_i
            left_ind, right_ind = split_dataset(X, i, thresholds[j])
            
            if sum(left_ind) < min_node_size || sum(right_ind) < min_node_size
                continue
            end

            l = L(X, y, i, thresholds[j])
            if l > best_split_val
                best_split_val = l
                best_feature = i
                if j < n_i
                    best_threshold = (thresholds[j] + thresholds[j + 1])/2
                else
                    best_threshold = thresholds[j]
                end
                best_left_ind = left_ind
                best_right_ind = right_ind
            end
        end
    end

    return best_left_ind, best_right_ind, best_feature, best_threshold, best_split_val
end



