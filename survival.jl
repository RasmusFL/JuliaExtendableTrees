# helper functions for the survival setup
#------------------------------------------------------------------------------------

# function for computing the t, Y and d vector
function survival_helper(y::Matrix{Float64})
    t = sort(unique(y[y[:, 2] .== 1, :][:, 1]))     # select the (unique) true survival times from y
    N = length(t)                                   # the number of true deaths in the sample
    n = size(y, 1)

    # compute the vectors of individuals at risk, Y, and the number of deaths, d
    Y = Array{Int}(undef, N)
    d = Array{Int}(undef, N)
    for i in 1:N
        Y_res::Int = 0
        d_res::Int = 0
        for l in 1:n
            if y[l, 1] >= t[i]
                Y_res += 1
            end
            if y[l, 1] == t[i] && y[l, 2] == 1
                d_res += 1
            end
        end
        Y[i] = Y_res
        d[i] = d_res
    end

    return(t, Y, d)
end

function Nelson_Aalen(y::Matrix{Float64})::Matrix{Float64}
    t, Y, d = survival_helper(y)
    N = length(t)                       # number of unique death times

    # if no observed deaths, return the zero function
    if N == 0
        return(hcat(0.0, 0.0))
    end

    chf = Array{Float64}(undef, N)
    chf[1] = d[1]/Y[1]
    for i in 2:N
        if Y[i - 1] != 0
            chf[i] = chf[i - 1] + d[i]/Y[i]
        else
            chf[i] = chf[i - 1]
        end
    end

    return(hcat(t, chf))
end

# computes the Nelson-Aalen estimator in the point time
function Nelson_Aalen_value(time::Float64, NAmatrix::Matrix{Float64})::Float64
    N = size(NAmatrix, 1)
    if time < NAmatrix[1, 1]
        return(0.0)
    end

    for i in 2:N
        if time < NAmatrix[i, 1]
            return(NAmatrix[i - 1, 2])
        end
    end

    NAmatrix[N, 2]
end

# the Kaplan-Meyer estimator directly from the Nelson-Aalen estimator
function Kaplan_Meyer(NAmatrix::Matrix{Float64})::Matrix{Float64}
    diff_NA = size(NAmatrix, 1) - 1     # the number of jumps of the Nelson-Aalen estimator
    KM = Array{Float64}(undef, diff_NA + 1)
    KM[1] = 1

    # compute the product integral of the NA step function
    for i in 2:(diff_NA + 1)
        KM[i] = KM[i - 1] * (1 - (NAmatrix[i, 2] - NAmatrix[i - 1, 2]))
    end

    return(hcat(NAmatrix[:, 1], KM))
end

# function for computing t, Y_1, Y_2, d_1, d_2
function survival_criteria_helper(X::Matrix{Float64}, y::Matrix{Float64}, feature::Int, threshold::Float64)
    t = sort(unique(y[y[:, 2] .== 1, :][:, 1]))     # select the (unique) true survival times from y
    N = length(t)                                   # the number of true deaths in the sample
    n = size(y, 1)

    # compute the vectors of individuals at risk, Y, and the number of deaths, d for each node
    Y1 = Array{Int}(undef, N)
    Y2 = Array{Int}(undef, N)
    d1 = Array{Int}(undef, N)
    d2 = Array{Int}(undef, N)
    for i in 1:N
        Y1_res::Int = 0
        Y2_res::Int = 0
        d1_res::Int = 0
        d2_res::Int = 0
        for l in 1:n
            if y[l, 1] >= t[i]
                if X[l, feature] <= threshold
                    Y1_res += 1
                else
                    Y2_res += 1
                end
            end
            if y[l, 1] == t[i] && y[l, 2] == 1
                if X[l, feature] <= threshold
                    d1_res += 1
                else
                    d2_res += 1
                end
            end
        end
        Y1[i] = Y1_res
        Y2[i] = Y2_res
        d1[i] = d1_res
        d2[i] = d2_res
    end

    return(t, Y1, d1, Y2, d2)
end