
using DynamicDiscreteSamplers, Chairmarks, Random

function setup(rng, indices)
    ds = DynamicDiscreteSampler()
    for i in indices
	push!(ds, i, (10^200)*rand(rng))
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
   for i in 1:n
	j = rand(rng, 1:n)
	delete!(ds, j)
	inds[i] = rand(rng, ds)
	push!(ds, j, (10^200)*rand(rng))
   end
   return ds
end

rng = Xoshiro(42)
for s in [10^i for i in 3:8]
    b1 = @b setup($rng, 1:$s) sample_fixed($rng, _, $s)
    b2 = @b setup($rng, 1:$s) sample_variable($rng, _, $s)
    println(b1.time/s*10^9, " ", b2.time/s*10^9,)
end
