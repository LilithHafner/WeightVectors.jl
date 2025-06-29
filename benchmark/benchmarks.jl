using WeightVectors
using WeightVectors: FixedSizeWeightVector, WeightVector, SemiResizableWeightVector # compat with older versions of WeightVectors.jl
using Random
include("../test/DynamicDiscreteSampler.jl")

rng = Xoshiro(42)
t0 = time()
ds = DynamicDiscreteSampler(rng)
push!(ds, 1, rand(rng))
push!(ds, 2, rand(rng))
x = rand(ds) + rand(ds)
y = rand(ds, 1000)
t1 = time()

using ChairmarksForAirspeedVelocity

"Convert a vector into a format that AirspeedVelocity can understand"
vector_to_trial(v) = ChairmarksForAirspeedVelocity.BenchmarkTools.Trial(ChairmarksForAirspeedVelocity.BenchmarkTools.Parameters(seconds=0,samples=length(v),evals=1,overhead=0,gctrial=false,gcsample=false),1e9v,zeros(length(v)),0,0)
constant(n) = ChairmarksForAirspeedVelocity.Runnable(Returns(vector_to_trial([n, n])))

# SUITE is a magic global variable that AirspeedVelocity looks for
SUITE = BenchmarkGroup()

SUITE["TTFX excluding time to load"] = constant(t1-t0)

SUITE["empty constructor"] = @benchmarkable DynamicDiscreteSampler()

function gaussian_weights_sequential_push(n, σ)
    ds = DynamicDiscreteSampler()
    for i in 1:n
        push!(ds, i, exp(σ*randn()))
    end
    ds
end

function rand_delete(ds, n)
    res = 0
    for i in 1:n
        j = rand(ds)
        delete!(ds, j)
        res = res >> 1 + j
    end
    res
end

function rand_update(ds, σ)
    j = rand(ds)
    delete!(ds, j)
    push!(ds, j, exp(σ*randn()))
    j
end

function intermixed_h(n, σ)
    ds = DynamicDiscreteSampler()
    elements = Set{Int}()
    res = 0
    for i in 1:n
        if rand() < 0.5
            element = rand(1:n)
            if element ∉ elements
                push!(ds, element, exp(σ*randn()))
                push!(elements, element)
            end
        elseif length(elements) > 0
            element = rand(elements)
            delete!(ds, element)
            delete!(elements, element)
        end
        if length(elements) > 0
            res += rand(ds)
        end
    end
    res
end

for n in [100, 1000, 10000], σ in [.1, 1, 10, 100]
    # TODO: try to use min over noise, average over rng, and max over treatment in analysis
    SUITE["constructor n=$n σ=$σ"] = @benchmarkable gaussian_weights_sequential_push($n, $σ)
    SUITE["sample n=$n σ=$σ"] = @benchmarkable gaussian_weights_sequential_push(n, σ) rand
    SUITE["delete ∘ rand n=$n σ=$σ"] = @benchmarkable gaussian_weights_sequential_push(n, σ) rand_delete(_, $n) evals=1
    SUITE["update ∘ rand n=$n σ=$σ"] = @benchmarkable gaussian_weights_sequential_push(n, σ) rand_update(_, $σ) evals=n
    SUITE["intermixed_h n=$n σ=$σ"] = @benchmarkable intermixed_h($n, $σ)
    SUITE["summarysize n=$n σ=$σ"] = ChairmarksForAirspeedVelocity.Runnable() do
        vector_to_trial([3600Base.summarysize(gaussian_weights_sequential_push(n, σ)) for _ in 1:1_000_000÷n])
    end
end

for n in [10^3, 10^6], k in [10^4, 10^6], σ in [1, 100]
    SUITE["sample (bulk) n=$n k=$k σ=$σ"] = @benchmarkable gaussian_weights_sequential_push(n, σ) rand(_, $k) seconds=1
end

function pathological1_setup()
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1e50)
    ds
end
function pathological1_update(ds)
    push!(ds, 2, 1e100)
    delete!(ds, 2)
end
SUITE["pathological 1"] = @benchmarkable pathological1_setup pathological1_update
function pathological1′_update(ds)
    push!(ds, 2, 1e100)
    delete!(ds, 2)
    rand(ds)
end
SUITE["pathological 1′"] = @benchmarkable pathological1_setup pathological1′_update

function pathological2_setup()
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1e-300)
    ds
end
function pathological2_update(ds)
    push!(ds, 2, 1e300)
    delete!(ds, 2)
end
SUITE["pathological 2"] = @benchmarkable pathological2_setup pathological2_update
function pathological2′_update(ds)
    push!(ds, 2, 1e300)
    delete!(ds, 2)
    rand(ds)
end
SUITE["pathological 2′"] = @benchmarkable pathological2_setup pathological2′_update
function pathological2′′_setup()
    ds = DynamicDiscreteSampler()
    for i in 3:10^5
        push!(ds, i, 1e-300)
    end
    ds
end
SUITE["pathological 2′′"] = @benchmarkable pathological2′′_setup pathological2′_update

pathological3 = DynamicDiscreteSampler()
push!(pathological3, 1, 1e300)
delete!(pathological3, 1)
push!(pathological3, 1, 1e-300)
SUITE["pathological 3"] = @benchmarkable pathological3 rand

function pathological4_setup()
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1e-270)
    ds
end
function pathological4_update(ds)
    push!(ds, 2, 1e307)
    delete!(ds, 2)
end
SUITE["pathological 4"] = @benchmarkable pathological4_setup pathological4_update
function pathological4′_update(ds)
    push!(ds, 2, 1e307)
    delete!(ds, 2)
    rand(ds)
end
SUITE["pathological 4′"] = @benchmarkable pathological4_setup pathological4′_update
function pathological4′′_setup()
    ds = DynamicDiscreteSampler()
    for i in 3:10^5
        push!(ds, i, 1e-270)
    end
    ds
end
SUITE["pathological 4′′"] = @benchmarkable pathological4′′_setup pathological4′_update

function pathological5a_setup()
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 2.0^-32)
    push!(ds, 2, 1.0)
    ds
end
function pathological5a_update(ds)
    push!(ds, 3, 2.0^18)
    delete!(ds, 3)
end
SUITE["pathological 5a"] = @benchmarkable pathological5a_setup pathological5a_update
function pathological5b_setup()
    ds = DynamicDiscreteSampler()
    for i in 128:-1:1
        push!(ds, i, 2.0^-i)
    end
    ds
end
function pathological5b_update(ds)
    push!(ds, 129, 2.0^30)
    delete!(ds, 129)
end
SUITE["pathological 5b"] = @benchmarkable pathological5b_setup pathological5b_update
function pathological5b′_update(ds)
    push!(ds, 129, 2.0^48)
    delete!(ds, 129)
    rand(ds)
end
SUITE["pathological 5b′"] = @benchmarkable pathological5b_setup pathological5b′_update
function pathological5b′′_update(ds)
    push!(ds, 129, 2.0^67)
    delete!(ds, 129)
    rand(ds)
end
function pathological5b′′_setup()
    ds = DynamicDiscreteSampler()
    for j in 1:10^3
        for i in 128:-1:1
            push!(ds, 128*j+i+1, 2.0^-i)
        end
    end
    ds
end
SUITE["pathological 5b′′"] = @benchmarkable pathological5b′′_setup pathological5b′′_update

function pathological_compaction_setup()
    w = FixedSizeWeightVector(2^20+1)
    w[1:2^19] .= 1
    w[2^19+1:2^20] .= 2
    pathological_compaction_update!(w)
    w
end
function pathological_compaction_update!(w)
    for i in 0:5
        w[end] = 2^i
    end
end
SUITE["pathological old compaction (6-op)"] = @benchmarkable pathological_compaction_setup pathological_compaction_update!

function pathological_tiny_compaction_setup()
    w = FixedSizeWeightVector(1)
    pathological_compaction_update!(w)
    w
end
function pathological_tiny_compaction_update!(w)
    for i in 1:6
        w[end] = 2^i
    end
end
SUITE["pathological tiny compaction (6-op)"] = @benchmarkable pathological_tiny_compaction_setup pathological_tiny_compaction_update!

function pathological_small_compaction_setup()
    FixedSizeWeightVector([1,1,1,1,2,2,2,2,4,4,8,8,8])
end
function pathological_small_compaction_update!(w)
    w[9] = 1
    w[10] = 2
    w[9] = 8
    w[10] = 8
    w[10] = 16
    w[11] = 32
    w[12] = 16
    w[13] = 32
    w[9] = 16
    w[11] = 1
    w[13] = 2
    w[11] = 16
    w[13] = 16
    w[11] = 8
    w[9] = 4
    w[10] = 4
    w[12] = 8
    w[13] = 8
end
SUITE["pathological small compaction (18-op)"] = @benchmarkable pathological_small_compaction_setup pathological_small_compaction_update!

function pathological_medium_compaction_setup()
    FixedSizeWeightVector(vcat(fill(1, 66), repeat(2.0 .^ (1:66), inner=128)))
end
function pathological_medium_compaction_update!(w)
    for i in 1:66
        w[i] = 2.0^i
    end
    for i in 1:18
        for j in 1:33
            w[2j-1] = 2.0^(-2i)
            w[2j] = 2.0^(-2i-1)
        end
    end
end
SUITE["pathological medium compaction (1254-op)"] = @benchmarkable pathological_medium_compaction_setup pathological_medium_compaction_update!


function pathological_large_compaction_setup()
    FixedSizeWeightVector(vcat(fill(1, 2^10+2), repeat(2.0 .^ (-2:2^10-1), inner=2^10)))
end
function pathological_large_compaction_update!(w)
    for i in 1:2^10+2
        w[i] = 2.0^(i-3)
    end
    for i in 2:130
        for j in 1:2^9+1
            w[2j-1] = 2.0^(-2i)
            w[2j] = 2.0^(-2i-1)
        end
    end
end
SUITE["pathological large compaction (133380-op)"] = @benchmarkable pathological_large_compaction_setup pathological_large_compaction_update!

include("code_size.jl")
_code_size = code_size(dirname(pathof(WeightVectors)))

SUITE["code size in lines"] = constant(3600_code_size.lines)
SUITE["code size in bytes"] = constant(3600_code_size.bytes)
SUITE["code size in syntax nodes"] = constant(3600_code_size.syntax_nodes)
