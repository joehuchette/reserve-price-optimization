const DCHyperParamConfig = NamedTuple{(:λ, :γ),Tuple{Float64,Float64}}

struct DifferenceOfConvex{T <: Algorithm} <: Algorithm
    optimizer_factory
    initialization_algorithm::T
    hyperparameter_set::Array{DCHyperParamConfig}
end

function DifferenceOfConvex(optimizer_factory, initialization_algorithm::T=RandomGuessing()) where {T <: Algorithm}
    return DifferenceOfConvex{T}(
        optimizer_factory,
        initialization_algorithm,
        [(λ = 2.0^λ_exp_val, γ = 10.0^γ_exp_val) for λ_exp_val in -5:5, γ_exp_val in -3:-1]
    )
end

function _get_initializer_learner(existing_learner::Learner{DifferenceOfConvex{T}}) where {T <: Algorithm}
    return Learner(
        existing_learner.train_data,
        existing_learner.test_data,
        existing_learner.validation_data,
        existing_learner.algorithm.initialization_algorithm,
        existing_learner.param_lb,
        existing_learner.param_ub,
        existing_learner.offset_lb,
        existing_learner.offset_ub,
        existing_learner.objective_penalties
    )
end

function δv(r, bid_price, pay_price, γ)
    if r < pay_price
        return -1.0
    elseif r > (1 + γ) * bid_price
        return 1.0
    else
        return 0.0
    end
end

function δV(dataset::DataSet, weights, γ)
    num_features = feature_count(dataset)
    num_samples = sample_count(dataset)
    grads = zeros(num_features)
    for sample in dataset.samples
        linear_value = dot(weights, sample.features)
        for i in 1:num_features
            grads[i] += sample.features[i] * δv(linear_value, sample.bid_price, sample.pay_price, γ)
        end
    end
    return grads
end

function DCA(learner::Learner{DifferenceOfConvex{T}}, w_prev::Vector{Float64}, λ::Float64, γ::Float64) where {T <: Algorithm}
    dataset = learner.train_data
    num_features = feature_count(learner.train_data)
    num_samples = sample_count(learner.train_data)
    jump_model = JuMP.Model(JuMP.with_optimizer(learner.algorithm.optimizer_factory))
    JuMP.@variable(jump_model, w[1:num_features])
    JuMP.@variable(jump_model, s[1:num_samples])
    JuMP.@objective(jump_model, Min, λ * sum(w[i]^2 for i in 1:num_features) + sum(s) - dot(δV(dataset, w_prev, γ), w))
    for i in 1:num_samples
        sample = dataset.samples[i]
        features = sample.features
        bid_price = sample.bid_price
        JuMP.@constraint(jump_model, s[i] ≥ - dot(w, features))
        JuMP.@constraint(jump_model, s[i] ≥ 1 / γ * (dot(w, features) - (1 + γ) * bid_price))
    end
    JuMP.optimize!(jump_model)
    @assert JuMP.primal_status(jump_model) == MOI.FEASIBLE_POINT
    return JuMP.value.(w)
end

function line_search(dataset::DataSet, u::Vector{Float64}, γ::Float64)
    best_r = 0.0
    best_r_reward = Inf
    for _sample in dataset.samples
        _ux_val = dot(u, _sample.features)
        _ux_val <= 0 && continue
        r = _sample.bid_price / _ux_val
        r_reward = 0.0
        for sample in dataset.samples
            ux_val = dot(u, sample.features)
            ux_val <= 0 && continue
            pay_price = sample.pay_price / ux_val
            bid_price = sample.bid_price / ux_val
            if r <= pay_price
                r_reward += -pay_price * ux_val
            elseif pay_price < r <= bid_price
                r_reward += -r * ux_val
            elseif bid_price < r <= (1 + γ) * bid_price
                r_reward += 1 / γ * (r - (1 + γ) * bid_price) * ux_val
            else
                @assert r > (1 + γ) * bid_price
            end
        end
        if r_reward < best_r_reward
            best_r = r
            best_r_reward = r_reward
        end
    end
    return best_r
end

function train(learner::Learner{DifferenceOfConvex{T}}) where {T <: Algorithm}
    num_features = feature_count(learner.train_data)
    num_samples = sample_count(learner.train_data)
    # Subtlety of algorithm: Presume that a fixed λ attains the maximum for
    # any choice of w_{t-1} (i.e. it can stay the same after restart)
    hyperparam_results = TrainingResults[]
    for (λ, γ) in learner.algorithm.hyperparameter_set
        _initialization_learner = _get_initializer_learner(learner)
        _initialization_results = train(_initialization_learner)
        _linear_model = _initialization_results.model
        _audit_weights(_initialization_learner, _linear_model.weights)
        linear_model = _linear_model
        max_iter = 1000
        for iter in 1:max_iter
            iis_init_reward = _evaluate(learner.train_data, linear_model).reward
            v = DCA(learner, linear_model.weights, λ, γ)
            dca_linear_model = LinearModel(v)
            iis_dca_reward = _evaluate(learner.train_data, dca_linear_model).reward
            linear_model = dca_linear_model
            u = v / norm(v)
            η_star = line_search(learner.train_data, u, γ)
            # DIFFERENT THAN IN MM PAPER: Update should be η * u, not η * v
            w = η_star * u
            ls_linear_model = LinearModel(w)
            iis_ls_reward = _evaluate(learner.train_data, ls_linear_model).reward
            # New model is not strictly better than old one
            if !(iis_ls_reward > iis_init_reward)
                break
            end
            linear_model = ls_linear_model
        end
        push!(hyperparam_results, TrainingResults(learner, linear_model))
    end
    best_validation_reward = -Inf
    best_validation_index = 0
    for (i, result) in enumerate(hyperparam_results)
        reward = result.final_validation_eval.reward
        if reward > best_validation_reward
            best_validation_reward = reward
            best_validation_index = i
        end
    end
    return hyperparam_results[best_validation_index]
end
