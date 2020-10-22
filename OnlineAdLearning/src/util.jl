const REWARD_TOLERANCE = 1e-6
const CONSTR_TOLERANCE = 1e-3

_approx_eq(x, y) = abs(x - y) < CONSTR_TOLERANCE
_approx_geq_zero(x) = x > -CONSTR_TOLERANCE
_strictly_geq_zero(x) = x > CONSTR_TOLERANCE
_approx_between(x, l, u) = (x > l - CONSTR_TOLERANCE) && (x < u + CONSTR_TOLERANCE)

function _verify_solution(data::DataSet,
                opt_model::OptModel{Float64},
                only_warn::Bool=true)
    linear_model = opt_model.linear_model
    y_values = opt_model.ys
    z_values = opt_model.zs
    trivially_suboptimal = false
    for (i, sample) in enumerate(data.samples)
        bid_price = sample.bid_price
        pay_price = sample.pay_price
        y_value = y_values[i]
        z_value = z_values[i,:]
        linear_value = dot(linear_model.weights, sample.features) + linear_model.offset
        function _error(description)
            error("Expected: $description.\n Actual: linear_value=$linear_value, y_value=$y_value, pay_price=$pay_price, bid_price=$bid_price, and z_value=$z_value")
        end
        function _warn(description)
            @warn "Expected: $description.\n Actual: linear_value=$linear_value, y_value=$y_value, pay_price=$pay_price, bid_price=$bid_price, and z_value=$z_value"
        end
        reporting_fn = only_warn ? _warn : _error
        if !_approx_geq_zero(y_value - min(0.0, bid_price))
            reporting_fn("y ≥ min(0, bid_price)")
        end
        if _approx_eq(z_value[1], 1.0)
            if !_approx_eq(y_value, pay_price)
                reporting_fn("z[1] = 1 and y = pay_price")
            end
            if linear_value > pay_price + CONSTR_TOLERANCE
                reporting_fn("z[1] = 1 and <β,x> ≤ pay_price")
            end
        elseif _approx_eq(z_value[2], 1.0)
            if !_approx_geq_zero(linear_value - y_value)
                reporting_fn("z[2] = 1 and y ≤ <β,x>")
            end
            if !_approx_between(y_value, pay_price, bid_price)
                reporting_fn("z[2] = 1 and y ∈ [pay_price,bid_price]")
            end
        elseif _approx_eq(z_value[3], 1.0)
            if !_approx_eq(y_value, 0.0)
                reporting_fn("z[3] = 1 and y = 0")
            end
            if !_approx_geq_zero(linear_value - bid_price)
                reporting_fn("z[3] = 1 and <β,x> > bid_price")
            end
        else
            # _error("sum(z) = 1")
        end
    end
end

function _audit_weights(learner::Learner, β_values::Vector{Float64})
    num_boundary = 0
    for i in 1:length(β_values)
        if _approx_eq(β_values[i], learner.param_lb[i]) || _approx_eq(β_values[i], learner.param_ub[i])
            num_boundary += 1
        end
    end
    if num_boundary > 0
        @warn "Solution attained on boundary ($num_boundary of $(length(β_values)) components). Enlarge feasible region and continue."
    end
    return nothing
end

function _unif_random(L::Vector{Float64}, U::Vector{Float64})
    n = length(L)
    @assert n == length(U)
    x = Vector{Float64}(undef, n)
    for i in 1:n
        l = L[i]
        u = U[i]
        if l == u
            x[i] = l
        else
            @assert l < u
            x[i] = rand(Uniform(l, u))
        end
    end
    return x
end

function _reward_function(sample::DataPoint, model::LinearModel)
    linear_value = evaluate(model, sample.features)
    if linear_value <= sample.pay_price
        return sample.pay_price
    elseif linear_value <= sample.bid_price + REWARD_TOLERANCE
        return linear_value
    else
        return 0.0
    end
end

function _grad_reward_function(sample::DataPoint, model::LinearModel)
    num_features = length(sample.features)
    linear_value = evaluate(model, sample.features)
    if linear_value <= sample.pay_price
        return zeros(Float64, num_features), 0.0
    elseif linear_value <= sample.bid_price + REWARD_TOLERANCE
        return vcat(sample.features), 1.0
    else
        return zeros(Float64, num_features), 0.0
    end
end

function _compute_reward(data::DataSet, model::LinearModel)
    val = 0.0
    for sample in data.samples
        val += _reward_function(sample, model)
    end
    return val
end

function _compute_expected_reward(data::DataSet, model::LinearModel)
    return _compute_reward(data, model) / sample_count(data)
end

_compute_expected_reward_if_test_data(data::Nothing, model::LinearModel) = NaN
function _compute_expected_reward_if_test_data(data::DataSet, model::LinearModel)
    return _compute_expected_reward(data, model)
end

function _compute_customer_rewards(data::DataSet, model::LinearModel)
    return [_reward_function(sample, model) for sample in data.samples]
end

function _proportion_sold(data::DataSet, model::LinearModel)
    y_values = _compute_customer_rewards(data, model)
    num_sold = count(_strictly_geq_zero, y_values)
    return num_sold / sample_count(data)
end

_proportion_sold_if_test_data(data::Nothing, model::LinearModel) = NaN
function _proportion_sold_if_test_data(data::DataSet, model::LinearModel)
    return _proportion_sold(data, model)
end

import Distributions

import LinearAlgebra.I

function normal_log_normal_data(num_features, num_samples; ρ=0.9, σ=0.1, α=0.0)

    c = (1 / sqrt(num_features)) * rand(Distributions.Normal(), num_features)
    uncorrelated_c = (1 / sqrt(num_features)) * rand(Distributions.Normal(), num_features)
    correlated_c = ρ * c + sqrt(1 - ρ^2) * uncorrelated_c
    X = [1 / sqrt(num_features) * rand(Distributions.Normal(), num_features) for i in 1:num_samples]

    datapoints = Vector{DataPoint}(undef, num_samples)
    for i in 1:num_samples
        c_dot_x = LinearAlgebra.dot(c, X[i])
        corr_c_dot_x = LinearAlgebra.dot(correlated_c, X[i])
        price_one = rand(Distributions.LogNormal(c_dot_x, σ * abs(c_dot_x)))
        price_two = rand(Distributions.LogNormal(corr_c_dot_x, σ * abs(corr_c_dot_x)))
        datapoints[i] = DataPoint(
            X[i],
            (1 + α) * max(price_one, price_two),
            (1 - α) * min(price_one, price_two)
        )
    end
    mean_bid_price = StatsBase.mean(dp.bid_price for dp in datapoints)
    for i in 1:num_samples
        datapoints[i].bid_price /= mean_bid_price
        datapoints[i].pay_price /= mean_bid_price
    end

    return DataSet(datapoints)
end
