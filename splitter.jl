# functions for determining the best split

# splits a feature dataset based on a feature and a threshold value c in the form
# of bitwise vectors where 1 indicates inclusion of the element and 0 not

# returns the indices of the left and right split
function split_dataset(X::Matrix{Float64}, feature::Int, threshold::Float64)
    left_ind = X[:, feature] .<= threshold
    right_ind = .!left_ind
    return left_ind, right_ind
end

# helper functions
#------------------------------------------------------------------------------------

# iterate through all unique values (thresholds) of the feature to find the best split
# it is assumed that high values of L indicate a better split
function best_feature_split(X::Matrix{Float64}, y::AbstractArray, L::Function, feature::Int, min_node_size::Int, thresholds::Array{Float64})
    best_threshold = nothing
    best_split_val = -Inf
    best_left_ind = nothing
    best_right_ind = nothing

    n = length(thresholds)
    mid = Int(ceil(n/2))

    # check the first half of the splits
    for j in 1:(mid - 1)

        left_ind, right_ind = split_dataset(X, feature, thresholds[mid - j])

        # if the split creates a node with too few data points, skip the split
        # entirely and all following splits (works because thresholds is sorted)
        if sum(left_ind) < min_node_size || sum(right_ind) < min_node_size
            break
        end

        l = L(X, y, left_ind, right_ind)
        if l > best_split_val
            best_threshold = (thresholds[mid - j] + thresholds[mid - j + 1])/2
            best_split_val = l
            best_left_ind = left_ind
            best_right_ind = right_ind
        end
    end

    # check the second half of the splits
    for j in mid:n

        left_ind, right_ind = split_dataset(X, feature, thresholds[j])

        # if the split creates a node with too few data points, skip the split
        # entirely and all following splits
        if sum(left_ind) < min_node_size || sum(right_ind) < min_node_size
            break
        end

        l = L(X, y, left_ind, right_ind)
        if l > best_split_val
            if j < n
                best_threshold = (thresholds[j] + thresholds[j + 1])/2
            else
                best_threshold = thresholds[j]
            end
            best_split_val = l
            best_left_ind = left_ind
            best_right_ind = right_ind
        end
    end
    
    # return the best split for this particular feature
    return best_left_ind, best_right_ind, best_threshold, best_split_val
end

# splitter for Classification
#------------------------------------------------------------------------------------

# finds the best split in the Classification regime
function best_split_Classification(X::Matrix{Float64}, y::Array{Int64}, L::Function, min_node_size::Int, n_features::Int, n_split::Int)
    best_feature = nothing
    best_threshold = nothing
    best_split_val = -1
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
        else
            thresholds = sort(thresholds)
        end

        best_left_ind_i, best_right_ind_i, best_threshold_i, best_split_val_i = best_feature_split(X, y, L, i, min_node_size, thresholds)

        if best_split_val_i > best_split_val
            best_feature = i
            best_threshold = best_threshold_i
            best_split_val = best_split_val_i
            best_left_ind = best_left_ind_i
            best_right_ind = best_right_ind_i
        end
    end

    # return the overall best split
    return best_left_ind, best_right_ind, best_feature, best_threshold, best_split_val
end

# splitter for Regression
#------------------------------------------------------------------------------------

# finds the best split in the Regression regime (same as in Classification)
function best_split_Regression(X::Matrix{Float64}, y::Array{Float64}, L::Function, min_node_size::Int, n_features::Int, n_split::Int)
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
        else
            thresholds = sort(thresholds)
        end

        best_left_ind_i, best_right_ind_i, best_threshold_i, best_split_val_i = best_feature_split(X, y, L, i, min_node_size, thresholds)

        if best_split_val_i > best_split_val
            best_feature = i
            best_threshold = best_threshold_i
            best_split_val = best_split_val_i
            best_left_ind = best_left_ind_i
            best_right_ind = best_right_ind_i
        end
    end

    # return the overall best split
    return best_left_ind, best_right_ind, best_feature, best_threshold, best_split_val
end

# splitter for Survival
#------------------------------------------------------------------------------------

# iterate through all unique values (thresholds) of the feature to find the best split for survival trees
# it is assumed that high values of L indicate a better split
function best_feature_split_Survival(X::Matrix{Float64}, y::Matrix{Float64}, L::Function, feature::Int, min_node_size::Int, thresholds::Array{Float64})
    best_threshold = nothing
    best_split_val = -1
    best_left_ind = nothing
    best_right_ind = nothing

    n = length(thresholds)
    mid = Int(ceil(n/2))

    # check the first half of the splits
    for j in 1:(mid - 1)

        left_ind, right_ind = split_dataset(X, feature, thresholds[mid - j])

        # if the split creates a node with too few data points, skip the split
        # entirely and all following splits (works because thresholds is sorted)
        if sum(left_ind) < min_node_size || sum(right_ind) < min_node_size
            break
        end

        l = L(X, y, feature, thresholds[mid - j])
        if l > best_split_val
            best_threshold = (thresholds[mid - j] + thresholds[mid - j + 1])/2
            best_split_val = l
            best_left_ind = left_ind
            best_right_ind = right_ind
        end
    end

    # check the second half of the splits
    for j in mid:n

        left_ind, right_ind = split_dataset(X, feature, thresholds[j])

        # if the split creates a node with too few data points, skip the split
        # entirely and all following splits
        if sum(left_ind) < min_node_size || sum(right_ind) < min_node_size
            break
        end

        l = L(X, y, feature, thresholds[j])
        if l > best_split_val
            if j < n
                best_threshold = (thresholds[j] + thresholds[j + 1])/2
            else
                best_threshold = thresholds[j]
            end
            best_split_val = l
            best_left_ind = left_ind
            best_right_ind = right_ind
        end
    end
    
    # return the best split for this particular feature
    return best_left_ind, best_right_ind, best_threshold, best_split_val
end

function best_split_Survival(X::Matrix{Float64}, y::Matrix{Float64}, L::Function, min_node_size::Int, n_features::Int, n_split::Int)
    best_feature = nothing
    best_threshold = nothing
    best_split_val = -1
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
        else
            thresholds = sort(thresholds)
        end

        best_left_ind_i, best_right_ind_i, best_threshold_i, best_split_val_i = best_feature_split_Survival(X, y, L, i, min_node_size, thresholds)

        if best_split_val_i > best_split_val
            best_feature = i
            best_threshold = best_threshold_i
            best_split_val = best_split_val_i
            best_left_ind = best_left_ind_i
            best_right_ind = best_right_ind_i
        end
    end

    # return the overall best split
    return best_left_ind, best_right_ind, best_feature, best_threshold, best_split_val
end
