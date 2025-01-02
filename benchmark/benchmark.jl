using DynamicDiscreteSamplers, Chairmarks, Random, Statistics

function setup(rng, indices)
    ds = DynamicDiscreteSampler()
    for i in indices
        push!(ds, i, (10.0^200)*rand(rng))
    end
    return ds
end

# Sampling with Fixed Distribution

function sample_fixed(rng, ds, n)
    return rand(rng, ds, n)
end

# Sampling with Variable Distribution

function sample_variable(rng, ds, n)
    inds = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        j = rand(rng, 1:n)
        delete!(ds, j)
        inds[i] = rand(rng, ds)
        push!(ds, j, (10.0^200)*rand(rng))
    end
    return ds
end

rng = Xoshiro(42)
for s in [10^i for i in 1:8]
    b1 = @be setup($rng, 1:$s) sample_fixed($rng, _, $s) seconds=3
    b2 = @be setup($rng, 1:$s) sample_variable($rng, _, $s) seconds=3
    println(mean(x.time for x in b1.samples)/s*10^9, " ", mean(x.time for x in b2.samples)/s*10^9,)
end
