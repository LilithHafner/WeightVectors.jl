using DynamicDiscreteSamplers, ChairmarksForAirspeedVelocity

# SUITE is a magic global variable that AirspeedVelocity looks for
SUITE = BenchmarkGroup()

SUITE["empty constructor"] = @benchmarkable DynamicDiscreteSampler()

function gaussian_weights_sequential_push(n, σ)
    ds = DynamicDiscreteSampler()
    for i in 1:n
        push!(ds, i, exp(σ*randn()))
    end
    ds
end

for n in [100, 1000, 10000], σ in [.1, 1, 10, 100]
    SUITE["constructor n=$n σ=$σ"] = @benchmarkable n,σ gaussian_weights_sequential_push(_...)
    SUITE["sample n=$n σ=$σ"] = @benchmarkable gaussian_weights_sequential_push(n, σ) rand
end
