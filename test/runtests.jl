using DynamicDiscreteSamplers
using HypothesisTests
using Random
using Test
using Aqua
using StableRNGs
using StatsBase

@test DynamicDiscreteSamplers.DEBUG === true

@testset "Constructor from array" begin
    w = FixedSizeWeights([1.7,2.9])
    @test w isa FixedSizeWeights
    @test w == [1.7, 2.9]
end

function ensure_sampler_conforms_to_rng_api(source, domain)
    # Conventional usage
    let x = rand(source)
        @test x isa Int
        @test x in domain
    end

    let x = rand(source, 3)
        @test x isa Vector{Int}
        @test length(x) == 3
        @test all(i in domain for i in x)
    end

    let x = [3, 4, 5]
        @test rand!(x, source) === x
        @test all(i in domain for i in x)
    end

    let x = rand(source, 3, 4, 5)
        @test x isa Array{Int, 3}
        @test size(x) == (3, 4, 5)
        @test all(i in domain for i in x)
    end

    let x = rand(source, 3, 4, 5)
        @test x isa Array{Int, 3}
        @test all(i in domain for i in x)
    end

    let x = fill(0, 2, 2, 2, 2)
        @test rand!(x, source) === x
        @test x isa Array{Int, 4}
        @test all(i in domain for i in x)
    end

    # Advanced usage (See https://docs.julialang.org/en/v1/stdlib/Random/#rand-api-hook)
    for rng in [Random.default_rng(), Xoshiro(42), StableRNG(42), Random.MersenneTwister(42)]
        let sampler = Random.Sampler(rng, source)
            x = [rand(rng, sampler) for _ in 1:1000]
            @test all(xi isa Int for xi in x)
            @test all(xi in domain for xi in x)
            @test !allequal(x)
        end
        let sampler = Random.Sampler(rng, source, Val(1))
            x = rand(rng, sampler)
            @test x isa Int
            @test x in domain
        end
        let sampler = Random.Sampler(rng, source, Val(Inf))
            x = [rand(rng, sampler) for _ in 1:1000]
            @test all(xi isa Int for xi in x)
            @test all(xi in domain for xi in x)
            @test !allequal(x)
        end
    end
end
w = DynamicDiscreteSamplers.FixedSizeWeights(2)
w[1] = 1.0
w[2] = 2.0
ensure_sampler_conforms_to_rng_api(w, 1:2)

include("DynamicDiscreteSampler_tests.jl") # Indirect tests for an upstream usage/legacy API

# These tests are too slow:
if "CI" in keys(ENV)
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DynamicDiscreteSamplers, deps_compat=false)
        Aqua.test_deps_compat(DynamicDiscreteSamplers, check_extras=false)
    end
end

include("weights.jl")
