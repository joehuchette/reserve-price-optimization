using CSV, DataFrames, Random

include("common.jl")

const df = CSV.read("cleaned.csv")

num_samples = size(df, 1)

fp = open(result_path, "w")
println(fp, join(["instance", "algorithm", "solve_time", "is_reward", "is_proportion_sold", "os_reward", "os_proportion_sold"], ","))

bid_prices = select(df, :first_price)
pay_prices = select(df, :second_price)

df = select!(df, Not([:first_price, :second_price]))

num_features = size(df, 2)

samples = Vector{OnlineAdLearning.DataPoint}(undef, num_samples)
for i in 1:num_samples
    features = [df[i,j] for j in 1:num_features]
    pay_price = pay_prices[i,1]
    bid_price = bid_prices[i,1]
    @assert bid_price >= pay_price
    samples[i] = OnlineAdLearning.DataPoint(features, bid_price, pay_price)
end

num_total_samples = length(samples)

for iter in 1:num_instances
    Random.seed!(iter)
    global samples = samples[randperm(num_total_samples)]

    train_data = OnlineAdLearning.DataSet(samples[1:num_train_points])
    test_data  = OnlineAdLearning.DataSet(samples[(num_train_points)+1:(num_train_points+num_test_points)])
    validation_data = OnlineAdLearning.DataSet(samples[(num_train_points+num_test_points+1):(num_train_points+num_test_points+num_validation_points)])
    @assert length(test_data.samples) == num_test_points
    @assert length(test_data.samples) == num_validation_points

    for (algo_name, algo_obj) in ALGORITHMS
        learner = OnlineAdLearning.Learner(train_data, algo_obj; test_data=test_data, validation_data=validation_data)
        solve_time = @elapsed(results = OnlineAdLearning.train(learner))
        println(fp, join([iter,
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
