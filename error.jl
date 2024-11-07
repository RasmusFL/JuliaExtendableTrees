# functions for computing the error of decision trees and random forests

# Classification
#------------------------------------------------------------------------------------

# error functions for a single decision tree

# computes the error for a classification tree
# (X_test, y_test) is the test data
function error_Classification(tree::Tree, X_test::Matrix{Float64}, y_test::Array{Int64})::Float64
    n = size(X_test, 1)
    incorrect = 0
    for i in 1:n
        if predict(tree, X_test[i, :]) != y_test[i]
            incorrect += 1
        end
    end
    incorrect/n
end

# computes the error (misclassification rate and R^2) for a classification forest
# (X_test, y_test) is the test set, OOB = true means
# that out of bag estimation is applied
function error_Classification(forest::Forest, X_test::Matrix{Float64}, y_test::Array{Int64}, OOB::Bool = false)::Float64
    1 - mean(predict(forest, X_test, OOB) .== y_test)
end

function R2_Classification(forest::Forest, X_test::Matrix{Float64}, y_test::Array{Int64}, OOB::Bool = false)::Float64
    predicted = predict(forest, X_test, OOB)
    1 - mean(predicted .== y_test)/mean()
end

# functions for computing the OOB error and OOB R^2 when the whole dataset is supplied (i.e. the typical case)
# does not take ties into account and hence applies an approximation, which is slightly faster

function OOB_error_Classification(forest::Forest)::Float64
    predicted_values = OOB_predict_Classification(forest)
    1 - mean(predicted_values .== forest.y)
end

function OOB_R2_Classification()::Float64
    predicted_values = OOB_predict_Classification(forest)
    1 - (1 - mean(predicted_values .== forest.y))/(1 - mean(mode(forest.y) .== forest.y))
end

# Regression
#------------------------------------------------------------------------------------

# computes the error for a regression tree
# (X_test, y_test) is the test data
function error_Regression(tree::Tree, X_test, y_test)::Float64
    n = size(X_test, 1)
    ssq = 0
    for i in 1:n
        ssq += (predict(tree, X[i, :]) - y_test[i])^2
    end
    ssq/n
end

# computes the error (mean squared error and R^2) for a regression forest
# (X_test, y_test) is the test set
function error_Regression(forest::Forest, X_test::Matrix{Float64}, y_test::Array{Float64}, OOB::Bool = false)::Float64
    squared_error(predict(forest, X_test, OOB) - y_test)
end

function R2_Regression(forest::Forest, X_test::Matrix{Float64}, y_test::Array{Float64}, OOB::Bool = false)::Float64
    predicted = predict(forest, X_test, OOB)
    1 - squared_error(y_test - predicted)/squared_error(y_test - ones(Float64, length(predicted)) * mean(y_test))
end

# functions for computing the OOB error and OOB R^2 when the whole dataset is supplied (i.e. the typical case)
# does not take ties into account and hence applies an approximation, which is slightly faster

function OOB_error_Regression(forest::Forest)::Float64
    predicted = OOB_predict_Regression(forest)
    squared_error(predicted - forest.y)
end

function OOB_R2_Regression(forest::Forest)::Float64
    predicted = OOB_predict_Regression(forest)
    1 - squared_error(predicted - forest.y)/squared_error(ones(Float64, length(predicted)) * mean(forest.y) - forest.y)
end

# Survival
#------------------------------------------------------------------------------------

# general function for computing the C-index based on survival data y and outcomes
function Harrell_C(outcomes::Vector{Float64}, y::Matrix{Float64})
    # initialise numerator and denominator
    Concordance = 0
    Permissible = 0
    n = length(outcomes)

    for i in 1:n
        for j in (i + 1):n
            # if T_i < T_j and delta_i = 0, the pair is not comparable
            if y[i, 1] < y[j, 1] && y[i, 2] == 0
                continue
            end
            # similarly with i and j reversed
            if y[j, 1] < y[i, 1] && y[j, 2] == 0
                continue
            end
            # if T_i = T_j and delta_i = delta_j, the pair is also
            # considered incomparable
            if y[i, 1] == y[j, 1] && y[i, 2] == y[j, 2]
                continue
            end

            # if T_i < T_j and outcomes[i] > outcomes[j], the model
            # predicts correctly (similarly for i and j reversed), so
            # add 1 to Concordance
            # if outcomes[i] = outcomes[j], the model is indecisive,
            # so add 0.5 to Concordance
            if y[i, 1] < y[j, 1] && outcomes[i] > outcomes[j]
                Concordance += 1
            elseif y[j, 1] < y[i, 1] && outcomes[j] > outcomes[i]
                Concordance += 1
            elseif outcomes[i] == outcomes[j]
                Concordance += 0.5
            end

            Permissible += 1
        end
    end

    return Concordance/Permissible
end

# computes Harrell's C-index for a set of survival predictions (NA-estimators)
function Harrell_C(predicted, y_test::Matrix{Float64})::Float64
    # compute the outcomes (ensemble mortality)
    n = length(predicted)
    outcomes = Array{Float64}(undef, n)
    for i in 1:n
        outcomes[i] = sum(predicted[i][:,2])
    end

    Harrell_C(outcomes, y_test)
end

# computes Harrell's C-index for a single survival tree
# (X_test, y_test) is the test data
# if a list of predicted values is already computed, it can be supplied to save time
function Harrell_C(tree::Tree, X_test::Matrix{Float64}, y_test::Matrix{Float64}, predicted = nothing)::Float64
    if predicted === nothing
        predicted = predict(tree, X)
    end

    Harrell_C(predicted, y_test)
end

# computes Harrell's C-index for a survival forest using OOB data
# (X_test, y_test) is the test data
# if a list of predicted values is already computed, it can be supplied to save time
function Harrell_C(forest::Forest, X_test::Matrix{Float64}, y_test::Matrix{Float64}; OOB::Bool = false, predicted = nothing)::Float64
    # if predictions are not supplied, compute them from scratch
    if predicted === nothing
        predicted = predict(forest, X_test, OOB)
    end

    Harrell_C(predicted, y_test)
end

# computes E = 1 - C, C being Harrell's C-index
function error_Survival(tree::Tree, X_test::Matrix{Float64}, y_test::Matrix{Float64}; predicted = nothing)::Float64
    1 - Harrell_C(tree, X_test, y_test, predicted)
end

function error_Survival(forest::Forest, X_test::Matrix{Float64}, y_test::Matrix{Float64}; OOB::Bool = false, predicted = nothing)::Float64
    1 - Harrell_C(forest, X_test, y_test; OOB = OOB, predicted = predicted)
end

# functions for computing the OOB error when the whole dataset is supplied (i.e. the typical case)
# does not take ties into account and should thus only be considered a fast approximation.

function OOB_error_Survival(forest::Forest)
    predicted = OOB_predict_Survival(forest)
    1 - Harrell_C(predicted, forest.y)
end
