struct Clairvoyant <: Algorithm end

function train(learner::Learner{Clairvoyant})
    train_data = learner.train_data
    test_data = learner.test_data
    validation_data = learner.validation_data
    has_test_data = test_data !== nothing
    has_validation_data = validation_data !== nothing
    return TrainingResults(
        LinearModel(fill(NaN, feature_count(test_data)), NaN),
        Evaluation(
            1 / sample_count(train_data) * sum(dp.bid_price for dp in train_data.samples),
            1.0,
            nothing),
        has_test_data ? Evaluation(
            1 / sample_count(test_data) * sum(dp.bid_price for dp in test_data.samples),
            1.0,
            nothing) : nothing,
        has_validation_data ? Evaluation(
            1 / sample_count(validation_data) * sum(dp.bid_price for dp in validation_data.samples),
            1.0,
            nothing) : nothing,
        IterationResult[]
    )
end
