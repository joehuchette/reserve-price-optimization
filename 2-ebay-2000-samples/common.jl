include("../OnlineAdLearning/src/OnlineAdLearning.jl")

using JuMP, Gurobi
using Random

TIME_LIMIT = 5 * 60.0

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

const num_train_points = 2000
const num_test_points = 2000
const num_validation_points = 2000
const num_instances = 10
