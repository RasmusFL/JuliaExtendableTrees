# functions to compute predicted values for general trees and other helper functions
#------------------------------------------------------------------------------------

# computes the predicted value of a tree, x is a vector of features
function predict(tree::Tree, x::Array{Float64})
    current_node = tree.root
    while current_node.val === nothing   # not yet in a terminal node
        if x[current_node.feature] <= current_node.threshold
            current_node = current_node.left
        else
            current_node = current_node.right
        end
    end
    return current_node.val
end

# computes the predicted values for a tree where x is a matrix of features
function predict(tree::Tree, x::Matrix{Float64})
    n = size(x, 1)
    predicted = Array{Any}(undef, n)
    for i in 1:n
        predicted[i] = predict(tree, x[i, :])
    end
    predicted
end

# computes a boolean vector indicating which bootstrap sets do not contain x
# x must be one of the rows of X, the training data
function OOB_indices(forest::Forest, x::Array{Float64})
    B = length(forest.trees)

    # by default, x is not OOB
    OOB = falses(B)

    # find the indices of all rows equal to x in the original data
    x_ind = findall(row -> row == x, eachrow(forest.X))

    @threads for i in 1:B
        # if none of the indices in a bootstrap set is one, x is OOB
        if sum(forest.bootstrap_indices[:, i][x_ind]) == 0
            OOB[i] = true
        end
    end
    OOB
end

# general function to compute an array of the predicted values over
# all trees in a forest
# if OOB is set to true, we only take the predictions over the rows
# where x is OOB. 
function predicted_labels(forest::Forest, x::Array{Float64}, OOB::Bool = false)
    if OOB
        oob = OOB_indices(forest, x)
        trees = forest.trees[oob]
    else
        trees = forest.trees
    end

    n = length(trees)
    res = Array{Any}(undef, n)
    @threads for i in 1:n
        res[i] = predict(trees[i], x)
    end

    res
end

# in the following functions, if OOB is set to true, the function computes an 
# out-of-bag estimate (only makes sense if x is in the training set)

# prediction for Classification trees and forests
#------------------------------------------------------------------------------------

# function to predict for Classification forests (majority rule) for a
# single feature vector x
function predict_Classification(forest::Forest, x::Array{Float64}, OOB::Bool = false)
    predictions = predicted_labels(forest, x, OOB)
    if isempty(predictions) == false
        return(mode(predictions))
    else
        return(mode(forest.y))
    end
end

# function to return all OOB predictions (works faster than applying the
# above function for all x in X)
function OOB_predict_Classification(forest::Forest)
    n = size(forest.X, 1)
    predictions = Array{Int64}(undef, n)
    OOB_ind = transpose(.!forest.bootstrap_indices)

    @threads for i in 1:n
        # choose the trees where X[i, :] is OOB
        trees = forest.trees[OOB_ind[:, i]]
        m = length(trees)

        # compute the predictions over all the trees where X[i, :] is OOB
        temp_predictions = Array{Int64}(undef, m)
        for j in 1:m
            temp_predictions[j] = predict(trees[j], X[i, :])
        end

        # compute the final prediction for X[i, :] as the mode
        # if X[i, :] is not OOB for any dataset, set the prediction
        # to be the mode over the whole dataset
        if isempty(temp_predictions) == false
            predictions[i] = mode(temp_predictions)
        else
            predictions[i] = mode(forest.y)
        end
    end
    predictions
end

# prediction for Regression forests
#------------------------------------------------------------------------------------

# function to predict for Regression forests (mean)
function predict_Regression(forest::Forest, x::Array{Float64}, OOB::Bool = false)
    predictions = predicted_labels(forest, x, OOB)
    if isempty(predictions) == false
        return(mean(predictions))
    else
        return(mean(forest.y))
    end
end

# function to return all OOB predictions (works faster than applying the
# above function for all x in X)
function OOB_predict_Regression(forest::Forest)
    n = size(forest.X, 1)
    predictions = Array{Float64}(undef, n)
    OOB_ind = transpose(.!forest.bootstrap_indices)

    @threads for i in 1:n
        # choose the trees where X[i, :] is OOB
        trees = forest.trees[OOB_ind[:, i]]
        m = length(trees)

        # compute the predictions over all the trees where X[i, :] is OOB
        temp_predictions = Array{Float64}(undef, m)
        for j in 1:m
            temp_predictions[j] = predict(trees[j], X[i, :])
        end

        # compute the final prediction for X[i, :] as the mean
        # if X[i, :] is not OOB for any dataset, set the prediction
        # to be the mean over the whole dataset
        if isempty(temp_predictions) == false
            predictions[i] = mean(temp_predictions)
        else
            predictions[i] = mean(forest.y)
        end
    end
    predictions
end

# prediction for Survival forests
#------------------------------------------------------------------------------------

# [Consider another approach for survival forests. Another idea is to always compute
# predictions in terms of ALL event times in each node to make ensemble prediction
# faster. But then prediction for a single node may end up being quite slow which
# also slows down the fitting process.]

# functions to predict for Survival forests (ensemble Nelson-Aalen estimator)
function ensemble_NA(predictions, t::Array{Float64})
    N = length(t)
    B = length(predictions)
    jump_values = zeros(Float64, N, B)

    # compute the matrix of jump values in terms of the event times
    for j in 1:B
        val = predictions[j]
        last_value = 0
        for i in 1:N
            k = 1
            while k <= size(val, 1) && val[k, 1] <= t[i]
                if val[k, 1] == t[i]
                    last_value = val[k, 2]
                end
                k += 1
            end
            # if no val[k, 1] match the t[i], no jumps have happened yet
            # and so the jump_value is zero
            if k != 1
                jump_values[i, j] = last_value
            end
        end
    end

    average_values = mean(jump_values, dims = 2)
    hcat(t, average_values)
end

function predict_Survival(forest::Forest, x::Array{Float64}, t::Array{Float64}, OOB::Bool = false)
    ensemble_NA(predicted_labels(forest, x, OOB), t)
end

# function to return all OOB predictions (works faster than applying the
# above function for all x in X)
function OOB_predict_Survival(forest::Forest)
    n = size(forest.X, 1)
    predictions = Array{Matrix{Float64}}(undef, n)
    OOB_ind = transpose(.!forest.bootstrap_indices)
    
    # save the sorted unique event times for later
    t = sort(unique(forest.y[forest.y[:, 2] .== 1, :][:, 1]))

    @threads for i in 1:n
        # choose the trees where X[i, :] is OOB
        trees = forest.trees[OOB_ind[:, i]]
        m = length(trees)

        # compute the predictions over all the trees where X[i, :] is OOB
        temp_predictions = Array{Matrix{Float64}}(undef, m)
        for j in 1:m
            temp_predictions[j] = predict(trees[j], X[i, :])
        end

        # compute the final prediction for X[i, :] as the mode
        # if X[i, :] is not OOB for any dataset, set the prediction
        # to be the mode over the whole dataset
        if isempty(temp_predictions) == false
            predictions[i] = ensemble_NA(temp_predictions, t)
        else
            predictions[i] = ensemble_NA(forest.y, t)
        end
    end
    predictions
end

# compute predicted values of forests
#------------------------------------------------------------------------------------

# computes the predicted value of a forest, x is a single feature vector
function predict(forest::Forest, x::Vector; OOB::Bool = false)
    if forest.type == "Classification"
        return(predict_Classification(forest, x, OOB))
    elseif forest.type == "Regression"
        return(predict_Regression(forest, x, OOB))
    elseif forest.type == "Survival"
        t = sort(unique(forest.y[forest.y[:, 2] .== 1, :][:, 1]))
        return(predict_Survival(forest, x, t, OOB))
    else
        throw("Error: Type of tree not recognised")
    end
end

# computes the predicted value of a forest, x is a matrix, each row a feature vector
# uses multithreading
function predict(forest::Forest, x::AbstractMatrix; OOB::Bool = false)
    n = size(x, 1)
    res = Array{Any}(undef, n)
    if forest.type == "Classification"
        @threads for i in 1:n
            res[i] = predict_Classification(forest, x[i, :], OOB)
        end
    elseif forest.type == "Regression"
        @threads for i in 1:n
            res[i] = predict_Regression(forest, x[i, :], OOB)
        end
    elseif forest.type == "Survival"
        t = sort(unique(forest.y[forest.y[:, 2] .== 1, :][:, 1]))
        @threads for i in 1:n
            res[i] = predict_Survival(forest, x[i, :], t, OOB)
        end
    else
        throw("Error: Type of tree not recognised")
    end
    res
end
