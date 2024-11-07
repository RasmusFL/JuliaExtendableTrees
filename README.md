# JuliaExtendableTrees
A prototype Julia library for random forests for classification, regression and survival written for my master's thesis "Machine learning methods for survival and multi-state models", soon to be available on my website. 

The library works pretty well in terms of accuracy, but not in terms of speed. If you wish to do a proper analysis in the context of research, I would recommend alternatives such as the R packages `randomForestSRC` or `ranger`. The library also currently has the limitation that it cannot handle categorical data with more than two levels. Also, all data need to be representable as floats.

## Example of usage (survival trees)
To use the library, load the library by including the file `JuliaExtendableTrees.jl`
```Julia
include("JuliaExtendableTrees.jl")
```
We illustrate the use of the library on a survival data set, namely the `pbc` dataset. Start by loading the data and extracting the response and features.
```Julia
df = CSV.read("pbc.csv", DataFrame)
y = Matrix{Float64}(df[:, [:days, :status]])
X = Matrix{Float64}(df[:, Not([:days, :status])])
```
The following code grows a single decision tree using the `grow_tree` function. This function requires four arguments and has several optional arguments. One needs to supply the features, the response, the type and the splitting rule. We grow a survival tree with the log-rank splitting rule as follows.
```Julia
tree = grow_tree(X, y, "Survival", L_log_rank)
```
For a tree, one can further specify the minimum node size, the maximum depth of the tree, the number of split points used in a split and the number of features used at each split. If none of these are set manually, default values are used. As for splitting rules, the possibilities are `L_log_rank` (log-rank), `L_conserve` (conservation of events splitting), `L_log_rank_score` (log-rank score), `L_approx_log_rank` (approximate log-rank) and `L_C` (C-index splitting).  Below is code for growing a tree with each hyperparameter specified manually (in this case to the default values).
```Julia
tree = grow_tree(X, y, "Survival", L_log_rank; min_node_size = 15, max_depth = 0,
			     n_split = 0, n_features = Int(round(sqrt(size(X, 2)))))
```
One can print information about the fitted tree to the terminal using the `print_tree` function. This function returns a vector of strings with information about each node plus some information about the tree itself. 

```Julia
print_tree(tree)
```
For every node, the depth is printed. If the node is not terminal, the feature index and the threshold value is provided. If the node is terminal, the value (in this case the Nelson--Aalen estimator) is printed along with the number of observations. We now turn to prediction. The following code predicts the Nelson--Aalen estimator based on the first observation in the training data.
```Julia
predict(tree, X[1, :])
```
This yields a matrix with the first column the jump times and the second column the corresponding value of the cumulative hazard function. Computing the predicted value of a whole dataset is easy. To illustrate, if one wants to compute all predictions of the training data, one simply writes
```Julia
predict(tree, X)
``` 
Computing an error estimate (the C-index error) is done in the following way. The `error_Survival` function needs a tree or forest as its first input, a test dataset as its second input and a test response as its third output. In code, this looks as follows.
```Julia
error_Survival(tree, X, y)
```
We now turn to forests. Growing a forest is very similar to a tree, although more hyperparameters are available. If one wants to use default parameters and the log-rank splitting rule, one can simply write the following.
```Julia
forest = grow_forest(X, y, "Survival", L_log_rank)
```
To obtain useful information on this forest, use the `print_forest` function,
```Julia
print_forest(forest)
```
This writes the type of forest ("Survival"), the number of available observations and features as well as the chosen hyperparameters. The function also prints the average number of terminal nodes in the forest. Prediction works the same way as with trees, but more options are available. To predict on the first feature in the dataset, simply run `predict(forest, X[1, :])`, and `predict(forest, X)` computes all predictions of the dataset. For forests, one can also choose to compute the ensemble Nelson--Aalen estimator only over the OOB features. This is done by adding `OOB = true` as follows:
```Julia
predict(forest, X[1, :]; OOB = true)
```
To compute the error, several methods are available. The general function is `error_Survival` which works very much as for trees. To compute the training error, simply run
```Julia
error_Survival(forest, X, y)
```
If one wishes to compute the OOB error, simply run
```Julia
error_Survival(forest, X, y; OOB = true)
```
instead. When the whole dataset is supplied (as here), a faster alternative is to run
```Julia
OOB_error_Survival(forest)
```
This function uses the OOB indices computed when the forest is fitted and does not take ties into account. Hence this function should be considered as a fast approximation which is exact when no ties are present in the feature data. This is the case for `pbc`, so the two functions above should yield identical results.

