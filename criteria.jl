# functions for computing the value of splits (denoted by L)

include("survival.jl")

# criteria for Classification
#------------------------------------------------------------------------------------

# all L functions here must take the inputs X, y, left_ind, right_ind where
# (X, y) is the data, and left_ind, right_ind are the bit-indices for the
# data in the left and right node, respectively

# computes an array of occurences (used for computing the Gini coefficient and entropy)
function proportions(y::Array{Int})::Array{Float64}
    counts = Dict{eltype(y), Int}()
    for elem in y
        counts[elem] = get(counts, elem, 0) + 1
    end
    collect(values(counts))/length(y)
end

# Gini coefficient for the array y
function Gini_coefficient(y::Array{Int})::Float64
    prop = proportions(y)
    1 - sum(prop .^2)
end

function L_Gini_coefficient(X::Matrix{Float64}, y::Array{Int}, left_ind::BitArray{1}, right_ind::BitArray{1})::Float64
    y_left = y[left_ind]
    y_right = y[right_ind]
    Gini_coefficient(y) -(length(y_left) * Gini_coefficient(y_left) + length(y_right) * Gini_coefficient(y_right))/length(y)
end

# Entropy for the array y
function Entropy(y::Array{Int64})::Float64
    prop = proportions(y)
    -transpose(prop) * map(log2, prop)
end

function L_Entropy(X::Matrix{Float64}, y::Array{Int}, left_ind::BitArray{1}, right_ind::BitArray{1})::Float64
    y_left = y[left_ind]
    y_right = y[right_ind]
    Entropy(y) - (length(y_left) * Entropy(y_left) + length(y_right) * Entropy(y_right))/length(y)
end

# criteria for Regression
#------------------------------------------------------------------------------------

# all L functions here must take the inputs X, y, left_ind, right_ind where
# (X, y) is the data, and left_ind, right_ind are the bit-indices for the
# data in the left and right node, respectively

# squared error for the array y
function squared_error(y::Array{Float64})::Float64
    sum(y .^2)/length(y) - (sum(y)/length(y))^2
end

function L_squared_error(X::Matrix{Float64}, y::Array{Float64}, left_ind::BitArray{1}, right_ind::BitArray{1})::Float64
    y_left = y[left_ind]
    y_right = y[right_ind]
    squared_error(y) - (length(y_left) * squared_error(y_left) + length(y_right) * squared_error(y_right))/length(y)
end

# absolute error for the array y
function abs_error(y::Array{Float64})::Float64
    sum(map(abs, y))/length(y)
end

function L_abs_error(X::Matrix{Float64}, y::Array{Float64}, left_ind::BitArray{1}, right_ind::BitArray{1})::Float64
    y_left = y[left_ind]
    y_right = y[right_ind]
    abs_error(y) - (length(y_left) * abs_error(y_left) + length(y_right) * abs_error(y_right))/length(y)
end

# criteria for Survival
#------------------------------------------------------------------------------------

# all L functions here must take the inputs X, y, feature, threshold where
# (X, y) is the data, feature is the index of the feature being split upon
# and threshold is the value of the threshold for the feature

# log-rank error
function L_log_rank(X::Matrix{Float64}, y::Matrix{Float64}, feature::Int, threshold::Float64)::Float64
    t, Y1, d1, Y2, d2 = survival_criteria_helper(X, y, feature, threshold)
    Y = Y1 + Y2
    d = d1 + d2

    sum_num = 0
    sum_den = 0
    for i in 1:length(t)
        if Y[i] < 2 || Y1[i] < 1
            break
        end
        if d[i] > 0
            sum_num += d1[i] - Y1[i] * d[i] / Y[i]
            sum_den += d[i] * (Y1[i] / Y[i]) * (1 - Y1[i]/Y[i]) * (Y[i] - d[i]) / (Y[i] - 1)
        end
    end

    if sum_den != 0
        return(abs(sum_num/sqrt(sum_den)))
    else
        return(-1)
    end
end

# conservation of events error
function L_conserve(X::Matrix{Float64}, y::Matrix{Float64}, feature::Int, threshold::Float64)::Float64
    t, Y1, d1, Y2, d2 = survival_criteria_helper(X, y, feature, threshold)
    N = length(t)
    if N == 0
        return(-1)
    end

    NA_sum1 = Array{Float64}(undef, N)
    NA_sum2 = Array{Float64}(undef, N)

    # compute vectors containing the innermost sums
    if d1[1] > 0
        NA_sum1[1] = d1[1]/Y1[1]
    end
    if d2[1] > 0
        NA_sum2[1] = d2[1]/Y2[1]
    end
    for l in 2:(N - 1)
        if d1[l] > 0
            NA_sum1[l] = NA_sum1[l - 1] + d1[l]/Y1[l]
        else
            NA_sum1[l] = NA_sum1[l - 1]
        end
        if d2[l] > 0
            NA_sum2[l] = NA_sum2[l - 1] + d2[l]/Y2[l]
        else
            NA_sum2[l] = NA_sum2[l - 1]
        end
    end

    # compute the sums running from 1 to N - 1 for each daughter
    sum1 = 0
    sum2 = 0
    for k in 1:(N - 1)
        sum1 += (Y1[k] - Y1[k + 1]) * Y1[k + 1] * NA_sum1[k]
        sum2 += (Y2[k] - Y2[k + 1]) * Y2[k + 1] * NA_sum2[k]
    end

    conserve = (Y1[1] * sum1 + Y2[1] * sum2) / (Y1[1] + Y2[1])

    1/(1 + conserve)
end

# computes the vector (Gamma_1, ..., Gamma_n)
function Gamma(T::AbstractArray{<:Real})
    n = length(T)
    res = zeros(Int, n)
    for i in 1:n
        for j in 1:n
            if T[j] <= T[i]
                res[i] += 1
            end
        end
    end

    res
end

# log-rank score error
function L_log_rank_score(X::Matrix{Float64}, y::Matrix{Float64}, feature::Int, threshold::Float64)::Float64
    x = X[:, feature]                       # x is the feature being split upon
    n = length(x)
    perm = sortperm(hcat(y, x)[:, 3])
    data = hcat(y, x)[perm, :]              # the data (y, x) sorted by the values in x
    G = Gamma(data[:, 1])                   # vector of Gamma_1, ..., Gamma_n
    a = Array{Float64}(undef, n)

    # compute the ranks a_1, ..., a_n
    for l in 1:n
        a_res = data[l, 2]
        for k in G[l]
            a_res += data[k, 2]/(n - G[k] + 1)
        end
        a[l] = a_res
    end

    sum = 0
    n1::Int = 0
    for l in 1:n
        if data[l, 3] <= threshold
            sum += a[l]
            n1 += 1
        end
    end

    den = sqrt(n1 * (1 - n1/n) * var(a))
    if den != 0
        return(abs((sum - n1 * mean(a))/den))
    else
        return(-1)
    end
end

# approximate log-rank error
function L_approx_log_rank(X::Matrix{Float64}, y::Matrix{Float64}, feature::Int, threshold::Float64)::Float64
    t, Y1, d1, Y2, d2 = survival_criteria_helper(X, y, feature, threshold)
    D1 = sum(d1)
    D2 = sum(d2)

    # the numerator
    num = 0
    for i in 1:length(t)
        num += d1[i] - Y1[i] * (d1[i] + d2[i]) / (Y1[i] + Y2[i])
    end

    den = sqrt((D1 - num) * (D2 + num))
    if den != 0
        return(abs(sqrt(D1 + D2) * num / den))
    else
        return(-1)
    end
end

# Harrell C splitting
function L_C(X::Matrix{Float64}, y::Matrix{Float64}, feature::Int, threshold::Float64)
    Concordance_left = 0    # number of concordant pairs in the left node
    Concordance_right = 0   # number of concordant pairs in the right node
    Concordance_mixed = 0   # number of concordant pairs where the observations are from different nodes
    Permissible = 0
    n = size(X, 1)

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
            
            # compute the number of permissible pairs in the left node, right node
            # and the number of permissible pairs, where the observations are from
            # different nodes
            if y[i, 1] < y[j, 1]
                if X[i, feature] <= threshold && X[j, feature] <= threshold
                    Concordance_left += 1
                elseif X[i, feature] > threshold && X[j, feature] > threshold
                    Concordance_right += 1
                elseif X[i, feature] > threshold && X[j, feature] <= threshold
                    Concordance_mixed += 1
                end
            elseif y[j, 1] < y[i, 1]
                if X[i, feature] <= threshold && X[j, feature] <= threshold
                    Concordance_left += 1
                elseif X[j, feature] > threshold && X[i, feature] > threshold
                    Concordance_right += 1
                elseif X[j, feature] > threshold && X[i, feature] <= threshold
                    Concordance_mixed += 1
                end
            end
        
            Permissible += 1
        end
    end

    C = (Concordance_mixed + 0.5 * Concordance_left + 0.5 * Concordance_right)/Permissible
end
