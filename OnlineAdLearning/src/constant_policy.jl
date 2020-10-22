struct ConstantPolicy <: Algorithm end

function train(learner::Learner{ConstantPolicy})
    train_samples = learner.train_data.samples
    best_r = 0.0
    best_r_reward = -Inf
    for _sample in train_samples
        r = _sample.bid_price
        r_reward = 0.0
        for sample in train_samples
            pay_price = sample.pay_price
            bid_price = sample.bid_price
            if r <= pay_price
                r_reward += pay_price
            elseif pay_price < r <= bid_price
                r_reward += r
            else
                @assert r > bid_price
            end
        end
        if r_reward > best_r_reward
            best_r = r
            best_r_reward = r_reward
        end
    end
    linear_model = LinearModel(zeros(feature_count(learner.train_data)), best_r)
    return TrainingResults(learner, linear_model)
end
