struct RandomGuessing <: Algorithm end

function train(learner::Learner{RandomGuessing})
    num_samples = sample_count(learner.train_data)
    num_features = feature_count(learner.train_data)
    β_values = _unif_random(learner.param_lb, learner.param_ub)
    offset = rand(Uniform(learner.offset_lb, learner.offset_ub))
    _audit_weights(learner, β_values)
    return TrainingResults(learner, LinearModel(β_values, offset))
end
