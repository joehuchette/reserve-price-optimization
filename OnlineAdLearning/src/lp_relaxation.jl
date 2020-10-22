const LPHyperParamConfig = NamedTuple{(:M, :λ),Tuple{Float64,Float64}}

struct LPRelaxation <: Algorithm
    optimizer_factory
    regularizer::Union{Nothing,ObjectivePenalty}
    hyperparameter_set::Array{LPHyperParamConfig}
end
function LPRelaxation(optimizer_factory; regularizer=nothing)
    return LPRelaxation(
        optimizer_factory,
        regularizer,
        [(M = 2.0^M_exp_val, λ = λ_val) for λ_val in (0.0,), M_exp_val in -1:9]
    )
end

has_regularizer(algo::LPRelaxation) = algo.regularizer !== nothing
get_regularizer(algo::LPRelaxation) = algo.regularizer

function train(learner::Learner{LPRelaxation}; hypograph::Bool=false)
    formulation_fn = hypograph ? formulate_data_point_hypograph_big_m : formulate_data_point_graph_big_m
    if learner.algorithm.regularizer !== nothing
        _add_objective_penalty!(learner, learner.algorithm.regularizer)
    end
    hyperparam_results = Dict{Any,TrainingResults}()
    for (M, λ) in learner.algorithm.hyperparameter_set
        learner.param_lb .= -M
        learner.param_ub .= M
        learner.offset_lb = -M
        learner.offset_ub = M
        if λ > 0
            # TODO: Refactor so that this is less hacky
            push!(learner.objective_penalties, L2Regularizer(λ))
        end
        opt_model = formulate_big_m(learner, formulation_fn, relax_integrality=true)
        jump_model = opt_model.jump_model
        solved_opt_model = optimize!(learner, opt_model)
        @assert solved_opt_model.jump_model == jump_model
        primal_status = JuMP.primal_status(jump_model)
        if primal_status != MOI.FEASIBLE_POINT
            @error "MIP did not produce a feasible solution"
        end
        termination_status = JuMP.termination_status(jump_model)
        if termination_status != MOI.OPTIMAL
            @warn "LP relaxation not solved to optimality (status = $termination_status)"
        end
        _audit_weights(learner, solved_opt_model.linear_model.weights)
        hyperparam_results[(M, λ)] = TrainingResults(learner, solved_opt_model.linear_model, reward_bound=JuMP.objective_value(solved_opt_model.jump_model))
    end
    best_validation_reward = -Inf
    best_validation_key = nothing

    iter_results = IterationResult[]
    for (key, result) in hyperparam_results
        reward = result.final_validation_eval.reward
        if reward > best_validation_reward
            best_validation_reward = reward
            best_validation_key = key
        end
        push!(iter_results, IterationResult(result.final_is_eval, result.final_os_eval, nothing, Dict("key" => key, "validation_eval" => result.final_validation_eval)))
    end
    best_result = hyperparam_results[best_validation_key]
    best_result.iteration_evals = iter_results
    return best_result
end
