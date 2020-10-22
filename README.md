# Contextual Reserve Price Optimization in Auctions via Mixed Integer Programming

This repository is the official implementation of "Contextual Reserve Price Optimization in Auctions via Mixed Integer Programming".

## Requirements

You must download and install:
* Gurobi: https://www.gurobi.com/
* Julia v1.x.x: https://julialang.org/downloads/

Additionally, you must acquire and register a license for the Gurobi library.

Once Julia has been installed, to install all dependencies first ensure that the [``GUROBI_HOME`` environment variable is set](https://github.com/JuliaOpt/Gurobi.jl#installation) and then open a Julia session and run:
```jl
import Pkg
REQUIRED_PACKAGES = ["Gurobi", "JuMP", "Optim", "LineSearches", "MathOptInterface", "Distributions", "StatsBase", "CSV", "DataFrames"]
for package in REQUIRED_PACKAGES
    Pkg.add(package)
end
```

## Training

To train the models in the paper, run this command:

```sh
/path/to/julia/binary 1-artificial-data/driver.jl
/path/to/julia/binary 2-ebay-2000-samples/driver.jl
/path/to/julia/binary 3-ebay-5000-samples/driver.jl
```

These three commands train the models which are summarized in Tables 1, 2(a), and 2(b), respectively.

## Evaluation

To analyze the results, run:

```sh
/path/to/julia/binary 1-artificial-data/analysis.jl
/path/to/julia/binary 2-ebay-2000-samples/analysis.jl
/path/to/julia/binary 3-ebay-5000-samples/analysis.jl
```

These three commands produce the results reported in Tables 1, 2(a), and 2(b), respectively.
