include("common.jl")

using CSV
using StatsBase

df = CSV.read(result_path)

for (algo_name, algo_fn) in ALGORITHMS
    slice = df[df.algorithm .== algo_name, :]
    println("$algo_name")
    println("-" ^ length(algo_name))
    println("  * solve time:    ", mean(slice.solve_time), " ± ", 2*StatsBase.std(slice.solve_time)/sqrt(num_instances), " sec")
    println("  * is_reward:     ", mean(slice.is_reward), " ± ", 2*StatsBase.std(slice.is_reward)/sqrt(num_instances))
    println("  * is_prop_sold:  ", mean(slice.is_proportion_sold), " ± ", 2*StatsBase.std(slice.is_proportion_sold)/sqrt(num_instances))
    println("  * os_reward:     ", mean(slice.os_reward), " ± ", StatsBase.std(slice.os_reward))
    println("  * os_prop_sold:  ", mean(slice.os_proportion_sold), " ± ", 2*StatsBase.std(slice.os_proportion_sold)/sqrt(num_instances))
end

STATS = ["solve_time", "is_reward", "os_reward", "is_proportion_sold", "os_proportion_sold"]
max_algo_name_length = maximum(length(algo_name) for (algo_name,algo_fn) in ALGORITHMS)

for stat in STATS
    println("$stat")
    println("-" ^ length(stat))
    for (algo_name, algo_fn) in ALGORITHMS
        slice = df[df.algorithm .== algo_name, :]
        println(rpad(string(algo_name, ": "), max_algo_name_length+2), mean(slice[!, Symbol(stat)]), " ± ", 2*StatsBase.std(slice[!, Symbol(stat)])/sqrt(num_instances))
    end
    slice = df[df.algorithm .== "clairvoyant", :]
    println(rpad("clairv.:", max_algo_name_length+2), mean(slice[!, Symbol(stat)]), " ± ", 2*StatsBase.std(slice[!, Symbol(stat)])/sqrt(num_instances))
    println()
end
winners = Dict(algo_name => 0 for (algo_name,algo_fn) in ALGORITHMS)
winners["tie"] = 0
for instance in 1:num_instances
    slice = df[df.instance .== instance, :]
    best_os_reward = maximum(slice.os_reward)
    # NOTE: This excludes ties from the tally!
    if size(slice[slice.os_reward .== best_os_reward, :], 1) != 1
        winners["tie"] += 1
        continue
    end
    for (algo_name, algo_fn) in ALGORITHMS
        if maximum(slice[slice.algorithm .== algo_name, :os_reward]) == best_os_reward
            winners[algo_name] += 1
            # break
        end
    end
end
@show winners

gap(lb, ub) = (ub - lb) / abs(ub)

function gap_closed(cl, dc, mip)
    mip_gap = gap(mip, cl)
    dc_gap = gap(dc, cl)
    return 1 - mip_gap / dc_gap
end

is_gaps_closed = Float64[]
for instance in 1:num_instances
    slice = df[df.instance .== instance, :]
    mip_is = maximum(slice[slice.algorithm .== "mip", :].is_reward)
    dc_is = maximum(slice[slice.algorithm .== "dc", :].is_reward)
    cl_is = maximum(slice[slice.algorithm .== "clairvoyant", :].is_reward)
    push!(is_gaps_closed, gap_closed(cl_is, dc_is, mip_is))
end
@show is_gaps_closed .* 100
@show mean(is_gaps_closed) * 100

os_gaps_closed = Float64[]
for instance in 1:num_instances
    slice = df[df.instance .== instance, :]
    mip_os = maximum(slice[slice.algorithm .== "mip", :].os_reward)
    dc_os = maximum(slice[slice.algorithm .== "dc", :].os_reward)
    cl_os = maximum(slice[slice.algorithm .== "clairvoyant", :].os_reward)
    push!(os_gaps_closed, gap_closed(cl_os, dc_os, mip_os))
end
@show os_gaps_closed .* 100
@show mean(os_gaps_closed) * 100
