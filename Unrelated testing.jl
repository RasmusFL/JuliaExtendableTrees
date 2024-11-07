function average_step_functions_known_jumps(step_functions::Vector{Any}, jump_points::Vector{Float64})
    n = length(step_functions)
    m = length(jump_points)
    values_at_jumps = zeros(Float64, m, n)

    for j in 1:n
        sf = step_functions[j]
        last_value = sf[1, 2]
        k = 1
        for i in 1:m
            while k <= size(sf, 1) && sf[k, 1] <= jump_points[i]
                last_value = sf[k, 2]
                k += 1
            end
            values_at_jumps[i, j] = last_value
        end
    end

    # Compute the average values at the known jump points
    average_values = mean(values_at_jumps, dims=2)

    # Form the resulting step function
    average_step_function = hcat(jump_points, average_values)

    return average_step_function
end

function predict_Survival_ChatGPT(forest::Forest, x, OOB::Bool = false, X = nothing)
    predictions = predicted_labels(forest, x, OOB, X)
    t = sort(unique(forest.y[forest.y[:, 2] .== 1, :][:, 1]))

    return(average_step_functions_known_jumps_optimized(predictions, t))
end

function predict_ChatGPT(forest::Forest, x, OOB::Bool = false, X = nothing)
    n = size(x, 1)
    res = Array{Any}(undef, n)
    @threads for i in 1:n
        res[i] = predict_Survival_ChatGPT(forest, x[i, :], OOB, X)
    end
    res
end

function predict_Survival(forest::Forest, x, OOB::Bool = false, X = nothing)
    predictions = predicted_labels(forest, x, OOB, X)
    
    t = sort(unique(forest.y[forest.y[:, 2] .== 1, :][:, 1]))
    N = length(t)
    B = length(predictions)
    jump_values = zeros(Float64, m, n)

    for j in 1:B
        val = predictions[j]
        last_value = val[1, 2]
        k = 1
        for i in 1:m
            while k <= size(val, 1) && val[k, 1] <= t[i]
                last_value = val[k, 2]
                k += 1
            end
            jump_values[i, j] = last_value
        end
    end

    average_values = mean(values_at_jumps, dims=2)

    hcat(t, average_values)
end

function split_dataset(X, feature::Int, threshold::Real)
    left_indicator = X[:, feature] .<= threshold
    right_indicator = .!left_indicator
    return left_indicator, right_indicator
end

# example data
#X = [3 4; 3 7; 3 4; 3 19; 4 -1; 4 -3; 4 0; 1 2; 4 7]
#y = [10.3 1; 12 1; 8.7 0; 8.2 1; 6.9 1; 9.0 0; 12.1 0; 11.3 1; 12 1]

X = [0, 0, 0, 0, 0, 1, 1, 1]
y = [1 0; 3 1; 3 0; 4 1; 5 1; 2 1; 6 1; 7 0]

#X = [0, 0, 0, 1, 1]
#y = [1 0; 2 1; 4 1; 3 1; 5 1]

# function to compare the two expressions for conserve
function conserve_test(X, y, feature::Int, threshold)
    t, Y_1, d_1, Y_2, d_2 = survival_criteria_helper(X, y, feature, threshold)
    N = length(t)

    left_ind, right_ind = split_dataset(X, feature, threshold)
    n_1 = sum(left_ind)
    n_2 = sum(right_ind)

    # sorted observations in each node
    y_1_sorted = sortslices(y[left_ind, :], dims = 1, by = x -> x[1])
    y_2_sorted = sortslices(y[right_ind, :], dims = 1, by = x -> x[1])

    NAmatrix_1 = Nelson_Aalen(y_1_sorted)
    NAmatrix_2 = Nelson_Aalen(y_2_sorted)
    NAmatrix = Nelson_Aalen(y)

    # start by computing conserve according to the definition
    M_1 = Array{Float64}(undef, n_1)
    M_2 = Array{Float64}(undef, n_2)
    M_1[1] = Nelson_Aalen_value(y_1_sorted[1, 1], NAmatrix_1) - y_1_sorted[1, 2]
    M_2[1] = Nelson_Aalen_value(y_2_sorted[1, 1], NAmatrix_2) - y_2_sorted[1, 2]

    # compute the M vectors in the definition of conserve
    for k in 2:n_1
        M_1[k] = M_1[k - 1] + (Nelson_Aalen_value(y_1_sorted[k, 1], NAmatrix_1) - y_1_sorted[k, 2])
    end
    for k in 2:n_2
        M_2[k] = M_2[k - 1] + (Nelson_Aalen_value(y_2_sorted[k, 1], NAmatrix_2) - y_2_sorted[k, 2])
    end

    println("M_1 = ", M_1)
    println("M_2 = ", M_2)

    # compute conserve as given in the definition
    conserve_def = (Y_1[1] * sum(abs.(M_1)) + Y_2[1] * sum(abs.(M_2)))/(Y_1[1] + Y_2[1])

    # the inner sums in the lemma
    sum_1 = 0
    sum_2 = 0

    NA_vec_1 = Array{Float64}(undef, N - 1)
    NA_vec_2 = Array{Float64}(undef, N - 1)
    if Y_1[1] != 0
        NA_vec_1[1] = d_1[1]/Y_1[1]
    else
        NA_vec_1[1] = 0
    end
    if Y_2[1] != 0
        NA_vec_2[1] = d_2[1]/Y_2[1]
    else
        NA_vec_2[1] = 0
    end
    
    for k in 2:(N - 1)
        if Y_1[k] != 0
            NA_vec_1[k] = NA_vec_1[k - 1] + d_1[k]/Y_1[k]
        else
            NA_vec_1[k] = NA_vec_1[k - 1]
        end
        if Y_2[k] != 0
            NA_vec_2[k] = NA_vec_2[k - 1] + d_2[k]/Y_2[k]
        else
            NA_vec_2[k] = NA_vec_2[k - 1]
        end
    end

    for k in 1:(N - 1)
        sum_1 += (Y_1[k] - Y_1[k + 1]) * Y_1[k + 1] * NA_vec_1[k]
        sum_2 += (Y_2[k] - Y_2[k + 1]) * Y_2[k + 1] * NA_vec_2[k]
    end

    conserve_lemma = (Y_1[1] * sum_1 + Y_2[1] * sum_2)/(Y_1[1] + Y_2[1])
    
    return(conserve_def, conserve_lemma)
end

function conserve_test_new(X, y, feature::Int, threshold)
    left_ind, right_ind = split_dataset(X, feature, threshold)
    n_1 = sum(left_ind)
    n_2 = sum(right_ind)

    # sorted observations in each node
    y_1_sorted = sortslices(y[left_ind, :], dims = 1, by = x -> x[1])
    y_2_sorted = sortslices(y[right_ind, :], dims = 1, by = x -> x[1])

    # compute the inner sums
    Y_1, d_1 = survival_helper(y)[2:3]
    Y_2, d_2 = survival_helper(y)[2:3]
    
    sums_1 = zeros(Float64, n_1 - 1)
    sums_2 = zeros(Float64, n_2 - 1)
    m_1 = 0
    m_2 = 0

    # compute the sums for j = 1
    if y_1_sorted[1, 2] == 1
        m_1 += 1
        sums_1[1] = d_1[m_1]/Y_1[m_1]
    end
    for k in 2:(n_1 - 1)
        if y_1_sorted[k, 2] == 1
            m_1 += 1
            sums_1[k] = sums_1[k - 1] + d_1[m_1]/Y_1[m_1]
        else
            sums_1[k] = sums_1[k - 1]
        end
    end
    # compute the sums for j = 2
    if y_2_sorted[1, 2] == 1
        m_2 += 1
        sums_2[1] = d_2[m_2]/Y_2[m_2]
    end
    for k in 2:(n_2 - 1)
        if y_2_sorted[k, 2] == 1
            m_2 += 1
            sums_2[k] = sums_2[k - 1] + d_2[m_2]/Y_2[m_2]
        else
            sums_2[k] = sums_2[k - 1]
        end
    end

    return(sums_1, sums_2)

    # now compute the outer sums
    sum_1 = 0
    sum_2 = 0

    for k in 1:(n_1 - 1)
        sum_1 += (n_1 - k) * sums_1[k]
    end
    for k in 1:(n_2 - 1)
        sum_2 += (n_2 - k) * sums_2[k]
    end

    return((Y_1[1] * sum_1 + Y_2[1] * sum_2)/(Y_1[1] + Y_2[1]))
end

# does not return the same! Maybe recreate the example in the notes?
# check code for errors really thoroughly!

# a test of column-major ordering
#N::Int = 10^4
#A = fill(1.0, (N, N))

# 14.859196 seconds
#@time begin
#    for i in 1:N
#        for j in 1:N
#            A[i, j] = 0.0
#        end
#    end
#end

# 13.034354 seconds
#@time begin
#    for i in 1:N
#        for j in 1:N
#            A[j, i] = 0.0
#        end
#    end
#end

