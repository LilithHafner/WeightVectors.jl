import Pkg
Pkg.add(url="https://github.com/LilithHafner/ChairmarksForAirspeedVelocity.jl")
Pkg.activate(".")
using ChairmarksForAirspeedVelocity, Serialization, Statistics
dir = "results"
separator = "-SEP-"
results = Dict(split(x, separator) => deserialize(joinpath(dir, x, "results.jls")) for x in readdir(dir))
head_sha = ENV["HEAD_SHA"]
base_sha = ENV["BASE_SHA"]
# 1,2,3
# lin,mac,win
# head,base

ks = keys(results)
trials = sort!(unique(getindex.(ks, 1)))
oss = sort!(unique(getindex.(ks, 2)))
shas = unique(getindex.(ks, 3))
head_sha, base_sha = shas # TODO: delete me
@assert issetequal(shas, [head_sha, base_sha])

# 0.123,1.324,12.32,436.3,6332 ms

acceptable_units(x) = mapreduce(acceptable_units, ∩, x)
function acceptable_units(x::Number)
    if x < 1e-7 # 100ns, 0.1 μs
        [(1e-9, "ns")]
    elseif x < 1e-5 # 10_000 ns, 10 μs
        [(1e-9, "ns"), (1e-6, "μs")]
    elseif x < 1e-4 # 100μs, 0.1 ms
        [(1e-6, "μs")]
    elseif x < 1e-2 # 10_000 μs, 10 ms, .01s
        [(1e-6, "μs"), (1e-3, "ms")]
    elseif x < 2 # 2000 ms, 2 s
        [(1e-3, "ms"), (1.0, "s")]
    else
        [(1.0, "s")]
    end
end
evaluate_unit(x, unit) = mapreduce(Base.Fix2(evaluate_unit, unit), +, x)
evaluate_unit(x::Number, unit) = unit[1] <= x <= unit[1] * 1000 || unit[1] == 1.0
function pick_unit(x)
    units = acceptable_units(x)
    isempty(units) && return nothing
    units[findmax(Base.Fix1(evaluate_unit, x), units)[2]]
end
show_number(io, x, ::Nothing, _=nothing) = show_number(io, x, pick_unit(x), true)
function show_number(io, x, unit, show_unit=true)
    x /= unit[1]
    if x == 0
        print(io, '0')
    else
        sigfigs = x < 1 ? 3 : 4
        r = round(x, digits=max(1, ceil(Int, sigfigs-log10(x))))
        print(io, r)
    end
    if show_unit
        print(io, " ", unit[2])
    end
end
function show_comma_separated(io, x, unit)
    if unit === nothing
        unit = pick_unit(x)
    end
    allequal(x) && return show_number(io, x[1], unit)
    for i in firstindex(x):lastindex(x)-1
        show_number(io, x[i], unit, false)
        print(io, ", ")
    end
    show_number(io, last(x), unit)
end
function round_ratio(x)
    isinteger(x) && return string(Int(x))
    isinf(x) && return string(x)
    sigfigs = 3
    r = round(x, digits=max(1, ceil(Int, sigfigs-log10(x))))
    string(r)
end

clean_os(os) = endswith(os, "-latest") ? os[1:end-7] : os[1:end]

benchmark_names = sort!(collect(only(unique(keys.(values(results))))))
name_len = maximum(length, benchmark_names)

times(sha, name) = [1e-9median(median(results[[trial, os, sha]][name]).time for trial in trials) for os in oss]

open("comment.md", "w") do io
    println(io, "### Benchmark Results\n")
    println(io, "|", ' '^name_len, " | [base](", base_sha, ")  | [pr](", head_sha, ") | [pr](", head_sha, ")/[base](", base_sha, ") |")
    println(io, "|:", '-'^(name_len+1), "|:---:|:---:|:---:|")
    for name in benchmark_names
        print(io, "| ", name, ' '^(name_len-length(name)), " | ")
        base_times = times(base_sha, name)
        head_times = times(head_sha, name)
        unit = pick_unit([base_times, head_times])
        show_comma_separated(io, base_times, unit)
        print(io, " | ")
        show_comma_separated(io, head_times, unit)
        print(io, " | ")
        if allequal(head_times ./ base_times)
            print(io, round_ratio(head_times[1] / base_times[1]))
        else
            join(io, round_ratio.(head_times ./ base_times), ", ")
        end
        println(io, " |")
    end

    println(io, "\n")
    println(io, "<details><summary>Full results</summary>")
    println(io, "<p>\n")

    println(io, "Results are printed as \"mean, low, q1, median, q3, high\" unless all data-points are equal in which case that value is printed.\n")

    print(io, "| ", ' '^name_len, " | os")
    for sha in [base_sha, head_sha], t in trials
        print(io, " | ", sha, " Trial ", t)
    end
    println(io, " |")
    println(io, "|:", '-'^(name_len+1), "|:---:|", ":---:|"^(length(trials)*2))

    for name in benchmark_names, os in oss
        print(io, "| ", name, ' '^(name_len-length(name)), " | ", clean_os(os))
        for sha in [base_sha, head_sha]
            for trial in trials
                ts = results[[trial, os, sha]][name].times
                summary = 1e-9vcat(mean(ts), quantile(ts, [0, 0.25, 0.5, 0.75, 1]))
                print(io, " | ")
                show_comma_separated(io, summary, nothing)
            end
        end
        println(io, " |")
    end

    println(io, "\n</p>")
    println(io, "</details>")
end
