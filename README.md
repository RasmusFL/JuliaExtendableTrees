# JuliaExtendableTrees
A prototype Julia library for random forests for classification, regression and survival written for my master's thesis "Machine learning methods for survival and multi-state models", soon to be available on my website. 

The library works pretty well in terms of accuracy, but not in terms of speed. If you wish to do a proper analysis in the context of research, I would recommend alternatives such as the R packages `randomForestSRC` or `ranger`.

## Example of usage
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

