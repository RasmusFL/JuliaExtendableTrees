include("JuliaExtendableTrees.jl")

# example data
#X = [3.0 4; 3 7; 3 4; 3 19; 4 -1; 4 -3; 4 0; 1 2; 4 7]
#y = [10.3 1; 12 1; 8.7 0; 8.2 1; 6.9 1; 9.0 0; 12.1 0; 11.3 1; 12 1]
#y = [1, 1, 2, 1, 2, 2, 1, 1, 2]

# grow_forest(X, y, type::String, L, max_depth, min_node_size, n_features, n_split, n_trees, sfrac, swr)
#test_forest = grow_forest(X, y, "Survival", L_log_rank, 0, 1, 1, 5, 1000)

# tests with pbc/veteran/follic

df = CSV.read("pbc.csv", DataFrame)
y = Matrix{Float64}(df[:, [:days, :status]])
X = Matrix{Float64}(df[:, Not([:days, :status])])

# testing a single tree

tree = grow_tree(X, y, "Survival", L_log_rank)

tree = grow_tree(X, y, "Survival", L_log_rank; min_node_size = 15, max_depth = 0, n_split = 0, n_features = Int(round(sqrt(size(X, 2)))))

print_tree(tree)

predict(tree, X[1, :])

predict(tree, X)

error_Survival(tree, X, y)

# testing a whole forest

@time begin
    forest = grow_forest(X, y, "Survival", L_log_rank)
end

print_forest(forest)

predict(forest, X[1, :])

predict(forest, X[1, :]; OOB = true)

error_Survival(forest, X, y)

error_Survival(forest, X, y; OOB = true)

OOB_error_Survival(forest)

#predictions = OOB_predict(forest)
#error_Survival(forest, X, y; OOB = true, predicted = predictions)



# cox PH

#df.event = EventTime.(df.days, df.status .== 1)
#model = coxph(@formula(event ~ treatment + age + sex + ascites + hepatom + spiders + edema + bili + chol + albumin + copper + alk +
#sgot + trig + platelet + prothrombin + stage), df)

#event_times = sort(unique(pbc_forest.y[pbc_forest.y[:, 2] .== 1, :][:, 1]))
#oob_predictions = predict(pbc_forest, X, true)
#predictions = predict(pbc_forest, X)
#test = Harrell_C(pbc_forest, X, y)

# test with peakVO2

#df = CSV.read("peakVO2.csv", DataFrame)
#y = Matrix(df[:, [:ttodead, :died]])
#X = Matrix(df[:, Not([:ttodead, :died])])

#@time begin
#    peakVO2_forest = grow_forest(X, y, "Survival", L_log_rank_score, 10, 5, 5, 10, 1000, 0.7, false)
#end

#@time begin
#    Harrell_C(peakVO2_forest, X, y)
#end


# test with wine

#wine = CSV.read("wine.csv", DataFrame)
#y = wine[:, :quality]
#X = Matrix(wine[:, Not(:quality)])

#@time begin
#    test_tree = grow_tree_Classification(X, y, L_Entropy, 10, 10, 5, 10)
#end

# grow_forest(X, y, type::String, L, max_depth, min_node_size, n_features, n_split, n_trees, sfrac, swr)
#@time begin
#    wine_forest = grow_forest(X, y, "Classification", L_Gini_coefficient; max_depth = 10, min_node_size = 5, sfrac = 0.7, swr = false) 
#end

#@time begin
#    wine_forest = grow_forest(X, y, "Classification", L_Gini_coefficient, 10, 15, 4, 10, 1000, 0.7, true)
#end

#@time begin
#    error_Classification(wine_forest, X, y, true)
#end

#@time begin
#    OOB_error_Classification(wine_forest)
#end


# test with BostonHousing

# grow_forest(X, y, type::String, L, max_depth, min_node_size, n_features, n_split, n_trees, sfrac, swr)

#BostonHousing = CSV.read("BostonHousing.csv", DataFrame)
#y = BostonHousing[:, :medv]
#X = Matrix(BostonHousing[:, Not(:medv)])
#@time begin
#    BH_forest = grow_forest_Regression(X, y, L_squared_error, 0, 10, Int(round(sqrt(size(X, 2)))) + 1, 10, 1000, 0.632, false)
#end

#test_tree = grow_tree_Regression(X, y, L_squared_error, 0, 10, Int(round(sqrt(size(X, 2)))) + 1, 10)

#OOB_error_Regression(BH_forest)
#OOB_R2_Regression(BH_forest)