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
    push!(ds, 2, 1e308)
    delete!(ds, 2)
end
SUITE["pathological 4"] = @benchmarkable pathological4_setup pathological4_update
