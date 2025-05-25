using DynamicDiscreteSamplers, BenchmarkTools, Random, Statistics

function setup(rng, indices)
    ds = DynamicDiscreteSampler()
    for i in indices
        push!(ds, i, exp2(200)*randexp(rng))
    end
    return ds
end

# Sampling Fixed Distribution

function sample_fixed_dist(rng, ds, n)
    return rand(rng, ds, n)
end

# Sampling Variable Distribution with Fixed Range

function sample_variable_dist_fixed_range(rng, ds, n)
    inds = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        j = rand(rng, ds)
        delete!(ds, j)
        push!(ds, j, exp2(200)*randexp(rng))
        inds[i] = j
    end
    return inds
end

# Sampling Variable Distribution with Growing Range

function sample_variable_dist_growing_range(rng, ds, n)
    inds = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        push!(ds, n+i, exp2(200)*randexp(rng))
        inds[i] = rand(rng, ds)
    end
    return inds
end

# Sampling Variable Distribution with Shrinking Range

function sample_variable_dist_shrinking_range(rng, ds, n)
    inds = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        j = rand(rng, ds)
        delete!(ds, j)
        inds[i] = j
    end
    return inds
end

rng = Xoshiro(42)
times = [[],[],[],[]]
for s in [[10^i for i in 1:7]..., 8*10^7]
    b1 = @benchmark sample_fixed_dist($rng, ds, $s) setup=(ds=setup($rng, 1:$s)) evals=1 seconds=10
    b2 = @benchmark sample_variable_dist_fixed_range($rng, ds, $s) setup=(ds=setup($rng, 1:$s)) evals=1 seconds=10
    b3 = @benchmark sample_variable_dist_growing_range($rng, ds, $s) setup=(ds=setup($rng, 1:$s)) evals=1 seconds=10
    b4 = @benchmark sample_variable_dist_shrinking_range($rng, ds, $s) setup=(ds=setup($rng, 1:$s)) evals=1 seconds=10
    push!(times[1], mean(b1.times)/s)
    push!(times[2], mean(b2.times)/s)
    push!(times[3], mean(b3.times)/s)
    push!(times[4], mean(b4.times)/s)
end

using Plots

plot!(1:8, times[1], marker=:circle, label="fixed dist", ylabel="time per element (ns)", xlabel="size", xticks=(1:8, [["10^$i" for i in 1:7]..., "8*10^7"]))
plot!(1:8, times[2], marker=:square, label="variable dist fixed range")
plot!(1:8, times[3], marker=:utriangle, label="variable dist growing range")
plot!(1:8, times[4], marker=:dtriangle, label="variable dist shrinking range")

