abstract type Algorithm end
abstract type ObjectivePenalty end

has_regularizer(algorithm::Algorithm) = false
get_regularizer(algorithm::Algorithm) = error("Algorithm does not have regularizer.")

mutable struct Learner{A <: Algorithm}
    train_data::DataSet
    test_data::Union{Nothing, DataSet}
    validation_data::Union{Nothing, DataSet}
    algorithm::A
    param_lb::Vector{Float64}
    param_ub::Vector{Float64}
    offset_lb::Float64
    offset_ub::Float64
    objective_penalties::Vector{ObjectivePenalty}
end
function Learner(train_data::DataSet, algorithm::A; test_data=nothing, validation_data=nothing, objective_penalties=ObjectivePenalty[], bounding_diameter=10.0) where {A <: Algorithm}
    max_abs_val = _max_abs_val(train_data)
    param_lb = -bounding_diameter * max_abs_val
    param_ub = bounding_diameter * max_abs_val
    max_bid_price = maximum(dp.bid_price for dp in train_data.samples)
    offset_lb = -bounding_diameter * max_bid_price
    offset_ub = bounding_diameter * max_bid_price
    if has_regularizer(algorithm)
        push!(objective_penalties, get_regularizer(algorithm))
    end
    return Learner{A}(train_data, test_data, validation_data, algorithm, param_lb, param_ub, offset_lb, offset_ub, objective_penalties)
end

struct ProximalPenalty <: ObjectivePenalty
    β::Vector{Float64}
    λ::Float64
end

struct L1Regularizer <: ObjectivePenalty
    coefficient::Float64
end
L1Regularizer() = L1Regularizer(0.0)

struct L2Regularizer <: ObjectivePenalty
    coefficient::Float64
end
L2Regularizer() = L2Regularizer(0.0)

function _add_objective_penalty!(learner::Learner, objective_penalty::ObjectivePenalty)
    push!(learner.objective_penalties, objective_penalty)
    return nothing
end

struct LinearModel{T <: Union{JuMP.VariableRef, Float64}}
    weights::Vector{T}
    offset::T
end
LinearModel(weights::Vector{Float64}) = LinearModel(weights, 0.0)

evaluate(model::LinearModel, x::Vector{Float64}) = LinearAlgebra.dot(model.weights, x) + model.offset

struct Evaluation
    reward::Float64
    proportion_sold::Float64
    reward_bound::Union{Nothing, Float64}
end

_evaluate(::Nothing, model::LinearModel) = nothing
function _evaluate(dataset::DataSet, model::LinearModel, reward_bound=nothing)
    reward = _compute_expected_reward(dataset, model)
    proportion_sold = _proportion_sold(dataset, model)
    Evaluation(reward, proportion_sold, reward_bound)
end

struct IterationResult
    is_eval::Evaluation
    os_eval::Union{Nothing,Evaluation}
    batch_eval::Union{Nothing,Evaluation}
    meta
end
function IterationResult(is_eval::Evaluation, os_eval::Union{Nothing,Evaluation}, batch_eval::Union{Nothing,Evaluation})
    return IterationResult(is_eval, os_eval, batch_eval, nothing)
end

mutable struct TrainingResults
    model::LinearModel
    final_is_eval::Evaluation
    final_os_eval::Union{Nothing,Evaluation}
    final_validation_eval::Union{Nothing,Evaluation}
    iteration_evals::Vector{IterationResult}
end
function TrainingResults(learner::Learner, model::LinearModel; reward_bound=nothing, iteration_results=IterationResult[])
    final_is_eval = _evaluate(learner.train_data, model, reward_bound)
    final_os_eval = _evaluate(learner.test_data, model)
    final_validation_eval = _evaluate(learner.validation_data, model)
    return TrainingResults(model, final_is_eval, final_os_eval, final_validation_eval, iteration_results)
end
