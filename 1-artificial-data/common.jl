include("../OnlineAdLearning/src/OnlineAdLearning.jl")

using JuMP, Gurobi

const TIME_LIMIT = 3 * 60.0

const solver = JuMP.with_optimizer(Gurobi.Optimizer, TimeLimit=TIME_LIMIT)
const root_solver = JuMP.with_optimizer(Gurobi.Optimizer, TimeLimit=TIME_LIMIT, NodeLimit=1, Crossover=0)
const relaxation_solver = JuMP.with_optimizer(Gurobi.Optimizer, TimeLimit=TIME_LIMIT, Method=2, Crossover=0)
const dc_solver = JuMP.with_optimizer(Gurobi.Optimizer, TimeLimit=TIME_LIMIT, OutputFlag=0)

const ALGORITHMS = Dict(
    "cp" => OnlineAdLearning.ConstantPolicy(),
    "clairvoyant" => OnlineAdLearning.Clairvoyant(),
    "dc" => OnlineAdLearning.DifferenceOfConvex(dc_solver),
    "mip" => OnlineAdLearning.MIP(solver),
    "mip_root" => OnlineAdLearning.MIP(root_solver),
    "lp" => OnlineAdLearning.LPRelaxation(relaxation_solver),
    "ga" => OnlineAdLearning.GradientAscent()
)

const result_path = "result.csv"

const num_features = 50

const num_train_points = 1000
const num_validation_points = 5000
const num_test_points = 5000
const num_samples = num_train_points + num_validation_points + num_test_points

const num_instances = 3

const HYPERPARAMS = Dict(
    "baseline" => (ρ = 0.9, σ = 0.1, α = 0.1),
    "high-noise" => (ρ = 0.9, σ = 0.5, α = 0.1),
    "low-correlation" => (ρ = 0.5, σ = 0.1, α = 0.1),
    "low-margin" => (ρ = 0.9, σ = 0.1, α = 0.02)
)

const ORDERED_KEYS = ["baseline", "high-noise", "low-correlation", "low-margin"]
