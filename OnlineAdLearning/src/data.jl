mutable struct DataPoint
    features::Vector{Float64}
    bid_price::Float64
    pay_price::Float64
end

mutable struct DataSet
    samples::Vector{DataPoint}
end
function DataSet(X::Vector{Vector{T}}, bid_prices::Vector{R}, pay_prices::Vector{S}) where {T, R, S <: Real}
    if !(length(X) == length(bid_prices) == length(pay_prices))
        error("Data set with sample size mismatch.")
    end
    if isempty(X)
        error("Cannot instantiate a problem with no data points.")
    end
    num_features = length(first(X))
    for point in X
        if length(point) != num_features
            error("Data points with different numbers of features.")
        end
    end
    for i in 1:length(bid_prices)
        if bid_prices[i] < pay_prices[i]
            error("Cannot set bid price below pay price.")
        end
        if pay_prices[i] < 0
            error("Cannot have negative pay price.")
        end
    end
    return DataSet([DataPoint(X[i], bid_prices[i], pay_prices[i]) for i in 1:length(X)])
end

function sample_count(dataset::DataSet)
    return length(dataset.samples)
end

function feature_count(dataset::DataSet)
    return length(dataset.samples[1].features)
end

function _max_abs_val(dataset::DataSet)
    return [maximum(abs.(sample.features[i] for sample in dataset.samples)) for i in 1:feature_count(dataset)]
end

function split_data(dataset::DataSet, num_train_points, num_validation_points, num_test_points)
    samples = dataset.samples
    @assert length(samples) == num_train_points + num_validation_points + num_test_points
    return samples[1:num_train_points],
           samples[(num_train_points+1):(num_train_points+num_validation_points)],
           samples[(num_train_points+num_validation_points+1):(num_train_points+num_validation_points+num_test_points)]
end
