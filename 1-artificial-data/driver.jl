using CSV, DataFrames, Distributions, LinearAlgebra, Random

include("common.jl")

fp = open(result_path, "w")
println(fp, join(["experiment", "iter", "algorithm", "solve_time", "is_reward", "is_proportion_sold", "os_reward", "os_proportion_sold"], ","))

for hp_name in ORDERED_KEYS, iter in 1:num_instances
    hp_params = HYPERPARAMS[hp_name]
    dataset = OnlineAdLearning.normal_log_normal_data(num_features, num_samples, ρ=hp_params.ρ, σ=hp_params.σ, α=hp_params.α)
    samples = dataset.samples
    train_data      = OnlineAdLearning.DataSet(samples[1:num_train_points])
    validation_data = OnlineAdLearning.DataSet(samples[(num_train_points)+1:(num_train_points+num_validation_points)])
    test_data       = OnlineAdLearning.DataSet(samples[(num_train_points+num_validation_points+1):(num_train_points+num_validation_points+num_test_points)])
    @assert length(test_data.samples) == num_test_points

    for (algo_name, algo_obj) in ALGORITHMS
        learner = OnlineAdLearning.Learner(train_data, algo_obj; test_data=test_data, validation_data=validation_data)
        solve_time = @elapsed(results = OnlineAdLearning.train(learner))
        println(fp, join([hp_name,
                          iter,
                          algo_name,
                          solve_time,
                          results.final_is_eval.reward,
                          results.final_is_eval.proportion_sold,
                          results.final_os_eval.reward,
                          results.final_os_eval.proportion_sold], ","))
        flush(fp)
    end
end
close(fp)
