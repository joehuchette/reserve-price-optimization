include("common.jl")

using CSV
using StatsBase

df = CSV.read(result_path)

STATS = ["solve_time", "is_reward", "os_reward", "is_proportion_sold", "os_proportion_sold"]
max_algo_name_length = maximum(length(algo_name) for (algo_name,algo_fn) in ALGORITHMS)

gap(lb, ub) = (ub - lb) / abs(ub)

function gap_closed(cl, dc, mip)
    mip_gap = gap(mip, cl)
    dc_gap = gap(dc, cl)
    return 1 - mip_gap / dc_gap
end

for experiment in ORDERED_KEYS
    println("EXPERIMENT $experiment")
    exper_df = df[df.experiment .== experiment, :]
    for stat in STATS
        println("$stat")
        println("-" ^ length(stat))
        for (algo_name, algo_fn) in ALGORITHMS
            slice = exper_df[exper_df.algorithm .== algo_name, :]
            println(rpad(string(algo_name, ": "), max_algo_name_length+2), mean(slice[!, Symbol(stat)]), " ± ", 2*StatsBase.std(slice[!, Symbol(stat)])/sqrt(num_instances))
        end
        println()
    end

    is_gaps_closed = Float64[]
    for instance in 1:num_instances
        slice = exper_df[exper_df.iter .== instance, :]
        mip_is = maximum(slice[slice.algorithm .== "mip", :].is_reward)
        dc_is = maximum(slice[slice.algorithm .== "dc", :].is_reward)
        cl_is = maximum(slice[slice.algorithm .== "clairvoyant", :].is_reward)
        push!(is_gaps_closed, gap_closed(cl_is, dc_is, mip_is))
    end
    println()
    println("IS gap closed: ", mean(is_gaps_closed) * 100)
    # @show mean(is_gaps_closed) * 100

    os_gaps_closed = Float64[]
    for instance in 1:num_instances
        slice = exper_df[exper_df.iter .== instance, :]
        mip_os = maximum(slice[slice.algorithm .== "mip", :].os_reward)
        dc_os = maximum(slice[slice.algorithm .== "dc", :].os_reward)
        cl_os = maximum(slice[slice.algorithm .== "clairvoyant", :].os_reward)
        push!(os_gaps_closed, gap_closed(cl_os, dc_os, mip_os))
    end
    println()
    println("OS gap closed: ", mean(os_gaps_closed) * 100)
    # @show mean(os_gaps_closed) * 100

    println()
    println()
end
