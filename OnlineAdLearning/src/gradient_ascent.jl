import Optim, LineSearches

struct GradientAscent <: Algorithm end

function train(learner::Learner{GradientAscent})
    num_samples = sample_count(learner.train_data)
    β_init = _unif_random(learner.param_lb, learner.param_ub)
    offset_init = rand(Uniform(learner.offset_lb, learner.offset_ub))

    l1_coeff = 0.0
    l2_coeff = 0.0
    for penalty in learner.objective_penalties
        if penalty isa L1Regularizer
            @assert l1_coeff == 0.0
            l1_coeff = penalty.coefficient
        elseif penalty isa L2Regularizer
            @assert l2_coeff = 0.0
            l2_coeff = penalty.coefficient
        else
            error("Unrecognized type of regularizer")
        end
    end
    function l(x)
        β = x[1:(end-1)]
        offset = x[end]
        model = LinearModel(β, offset)
        return -1 * (1 / num_samples * sum(_reward_function(data_point, model) for data_point in learner.train_data.samples) +
            l1_coeff * sum(abs.(β)) +
            l2_coeff * sum(β.^2))
    end
    function ∇l(x)
        β = x[1:(end-1)]
        offset = x[end]
        model = LinearModel(β, offset)
        return -1 * (1 / num_samples * sum(vcat(_grad_reward_function(data_point, model)...) for data_point in learner.train_data.samples) +
            l1_coeff * vcat(sign.(β), 0.0) +
            2l2_coeff * vcat(β, 0.0))
    end

    results = Optim.optimize(l, ∇l, vcat(β_init, offset_init), Optim.GradientDescent(linesearch=LineSearches.StrongWolfe()), Optim.Options(show_trace=false, allow_f_increases=true, extended_trace=false), inplace=false)

    β_values = results.minimizer[1:(end-1)]
    offset_value = results.minimizer[end]

    return TrainingResults(learner, LinearModel(β_values, offset_value))
end
