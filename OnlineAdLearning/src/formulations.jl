function apply!(penalty::ProximalPenalty, model::JuMP.Model, β::Vector{JuMP.VariableRef}, ys::Vector{JuMP.VariableRef})
    β⁰ = penalty.β
    @assert length(β) == length(β⁰)
    λ = penalty.λ
    quad_penalty = JuMP.@expression(model, sum((β⁰[i] - β[i])^2 for i in 1:length(β)))
    return 1 / λ * quad_penalty
end

function apply!(penalty::L1Regularizer, model::JuMP.Model, β::Vector{JuMP.VariableRef}, ys::Vector{JuMP.VariableRef})
    num_vars = length(β)
    w = JuMP.@variable(model, [i=1:num_vars], lower_bound=0)
    for i in 1:num_vars
        JuMP.@constraint(model, w[i] ≥ β[i])
        JuMP.@constraint(model, w[i] ≥ -β[i])
    end
    return penalty.coefficient * sum(w)
end

function apply!(penalty::L2Regularizer, model::JuMP.Model, β::Vector{JuMP.VariableRef}, ys::Vector{JuMP.VariableRef})
    quad_penalty = JuMP.@expression(model, sum((β[i])^2 for i in 1:length(β)))
    return penalty.coefficient * quad_penalty
end

struct OptModel{T <: Union{JuMP.VariableRef, Float64}}
    jump_model::JuMP.Model
    linear_model::LinearModel
    vs::Vector{T}
    ys::Vector{T}
    zs::Matrix{T}
end

function optimize!(learner::Learner, opt_model::OptModel)
    JuMP.optimize!(opt_model.jump_model)
    primal_status = JuMP.primal_status(opt_model.jump_model)
    if primal_status != MOI.FEASIBLE_POINT
        @error "MIP did not produce a feasible solution"
    end
    termination_status = JuMP.termination_status(opt_model.jump_model)
    if termination_status != MOI.OPTIMAL
        @warn "MIP did not solve to optimality"
    end
    weight_values = JuMP.value.(opt_model.linear_model.weights)
    offset_value = JuMP.value(opt_model.linear_model.offset)
    v_values = JuMP.value.(opt_model.vs)
    y_values = JuMP.value.(opt_model.ys)
    z_values = JuMP.value.(opt_model.zs)
    linear_model_value = LinearModel(weight_values, offset_value)
    _audit_weights(learner, weight_values)
    return OptModel(opt_model.jump_model,
        linear_model_value,
        v_values,
        y_values,
        z_values)
end

function formulate_big_m(learner::Learner{T}, formulation_fn::Function=formulate_data_point_graph_big_m; relax_integrality=false) where {T <: Union{MIP,LPRelaxation}}
    model = JuMP.Model(learner.algorithm.optimizer_factory)
    train_data = learner.train_data
    num_features = feature_count(train_data)
    num_samples = sample_count(train_data)
    β = JuMP.@variable(model, [i=1:num_features], lower_bound=learner.param_lb[i], upper_bound=learner.param_ub[i])
    offset = JuMP.@variable(model, lower_bound=learner.offset_lb, upper_bound=learner.offset_ub)
    vs = Vector{JuMP.VariableRef}(undef, num_samples)
    ys = Vector{JuMP.VariableRef}(undef, num_samples)
    zs = Matrix{JuMP.VariableRef}(undef, num_samples, 3)
    for i in 1:num_samples
        sample = train_data.samples[i]
        features = sample.features
        v = JuMP.@variable(model)
        y = JuMP.@variable(model, lower_bound = 0, upper_bound = sample.bid_price)
        JuMP.@constraint(model, v == sum(features[k] * β[k] for k in 1:num_features) + offset)
        z = formulation_fn(model, v, y, sample.pay_price, sample.bid_price, M⁻(learner, features), M⁺(learner, features), relax_integrality)
        vs[i] = v
        ys[i] = y
        zs[i,:] = z
    end
    # ret_val = [formulation_fn(model, β, train_data.samples[i], L, U, relax_integrality)
            # for i in 1:num_samples]
    # ys = [ret_val[i][1] for i in 1:num_samples]
    # indicators = [ret_val[i][2] for i in 1:num_samples]
    obj = 1 / num_samples * sum(ys)
    for penalty in learner.objective_penalties
        obj -= apply!(penalty, model, β, ys)
    end
    JuMP.@objective(model, Max, obj)
    return OptModel(model, LinearModel(β, offset), vs, ys, zs)
end

M⁻(features, L, U) = sum(min.(L .* features, U .* features))
M⁺(features, L, U) = sum(max.(L .* features, U .* features))

function M⁻(learner, features)
    @assert (n = length(features)) == length(learner.param_lb) == length(learner.param_ub)
    lb = learner.offset_lb
    for i in 1:n
        lb += min(learner.param_lb[i] * features[i], learner.param_ub[i] * features[i])
    end
    return lb
end

function M⁺(learner, features)
    @assert (n = length(features)) == length(learner.param_lb) == length(learner.param_ub)
    ub = learner.offset_ub
    for i in 1:n
        ub += max(learner.param_lb[i] * features[i], learner.param_ub[i] * features[i])
    end
    return ub
end

function formulate_data_point_hypograph_big_m(model::JuMP.Model,
        v::JuMP.VariableRef,
        y::JuMP.VariableRef,
        pay_price::Float64,
        bid_price::Float64,
        l::Float64,
        u::Float64,
        relax_integrality::Bool)
    if pay_price != 0.0
        error("Hypograph formulation does not currently work with pay prices.")
    end
    z = if relax_integrality
        JuMP.@variable(model, lower_bound=0.0, upper_bound=1.0)
    else
        JuMP.@variable(model, binary=true)
    end
    JuMP.@constraints(model, begin
        y <= bid_price * z
        y <= v - l * (1 - z)
        v >= u * (1 - z)
        v <= bid_price + (u - bid_price) * (1 - z)
    end)
    return z
end

function formulate_data_point_graph_big_m(model::JuMP.Model,
    v::JuMP.VariableRef,
    y::JuMP.VariableRef,
    pay_price::Float64,
    bid_price::Float64,
    l::Float64,
    u::Float64,
    relax_integrality::Bool)
    z = (if relax_integrality
        JuMP.@variable(model, [1:3], lower_bound=0.0, upper_bound=1.0)
    else
        JuMP.@variable(model, [1:3], binary=true)
    end)
    JuMP.@constraint(model, sum(z) == 1)
    JuMP.@constraints(model, begin
        # D1 = {(x,y) : y == pay_price, v <= pay_price}
        # D2 = {(x,y) : y == v, pay_price <= v <= bid_price}
        # D3 = {(x,y) : y == 0, v >= bid_price}
        y <= pay_price * z[1] + bid_price * z[2] + 0 * z[3]
        y >= pay_price * z[1] + pay_price * z[2] + 0 * z[3]
        y <= v + (pay_price - l) * z[1] - bid_price * z[3]
        y >= v - u * z[3]
    end)
    return z
end
