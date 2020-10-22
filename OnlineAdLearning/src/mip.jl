const MIPHyperParamConfig = NamedTuple{(:M, :λ),Tuple{Float64,Float64}}

struct MIP <: Algorithm
    optimizer_factory
    hypograph::Bool
    regularizer::Union{Nothing,ObjectivePenalty}
    cp_warm_start::Bool
    hyperparameter_set::Array{MIPHyperParamConfig}
end
function MIP(
    optimizer_factory;
    hypograph::Bool=false,
    regularizer=nothing,
    cp_warm_start=true)
    return MIP(
        optimizer_factory,
        hypograph,
        regularizer,
        cp_warm_start,
        [(M = 2.0^M_exp_val, λ = λ_val) for λ_val in (0.0,), M_exp_val in -1:9]
    )
end

has_regularizer(algo::MIP) = algo.regularizer !== nothing
get_regularizer(algo::MIP) = algo.regularizer

function train(learner::Learner{MIP})
    @assert learner.validation_data !== nothing

    formulation_fn = learner.algorithm.hypograph ? formulate_data_point_hypograph_big_m : formulate_data_point_graph_big_m
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
        opt_model = formulate_big_m(learner, formulation_fn)
        if learner.algorithm.cp_warm_start
            contextless_train_data = DataSet([DataPoint(Float64[], dp.bid_price, dp.pay_price) for dp in learner.train_data.samples])
            cp_learner = Learner(contextless_train_data, nothing, nothing, MIP(learner.algorithm.optimizer_factory), Float64[], Float64[], learner.offset_lb, learner.offset_ub, ObjectivePenalty[])
            cp_opt_model = formulate_big_m(cp_learner)
            cp_jump_model = cp_opt_model.jump_model
            cp_solved_opt_model = optimize!(cp_learner, cp_opt_model)
            @assert cp_solved_opt_model.jump_model === cp_jump_model
            if JuMP.primal_status(cp_jump_model) == MOI.FEASIBLE_POINT
                for i in 1:feature_count(learner.train_data)
                    JuMP.set_start_value(opt_model.linear_model.weights[i], 0.0)
                end
                JuMP.set_start_value(opt_model.linear_model.offset, cp_solved_opt_model.linear_model.offset)
                for i in 1:sample_count(learner.train_data)
                    JuMP.set_start_value(opt_model.vs[i], cp_solved_opt_model.vs[i])
                    JuMP.set_start_value(opt_model.ys[i], cp_solved_opt_model.ys[i])
                    for j in 1:size(opt_model.zs, 2)
                        JuMP.set_start_value(opt_model.zs[i,j], cp_solved_opt_model.zs[i,j])
                    end
                end
            else
                @warn "CP warmstart did not produce a feasible solution"
            end
        end
        solved_opt_model = optimize!(learner, opt_model)
        _verify_solution(learner.train_data, solved_opt_model)

        hyperparam_results[(M, λ)] = TrainingResults(learner, solved_opt_model.linear_model, reward_bound=JuMP.objective_bound(solved_opt_model.jump_model))
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
