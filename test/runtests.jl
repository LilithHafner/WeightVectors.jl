using WeightVectors
using HypothesisTests
using Random
using Test
using Aqua
using StableRNGs
using StatsBase

@test WeightVectors.DEBUG === true
@test_throws ErrorException WeightVectors.@fail 0

@testset "Constructor from array" begin
    w = FixedSizeWeightVector([1.7,2.9])
    @test w isa FixedSizeWeightVector
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
w = WeightVectors.FixedSizeWeightVector(2)
w[1] = 1.0
w[2] = 2.0
ensure_sampler_conforms_to_rng_api(w, 1:2)

@testset "unsigned int construction" begin
    @test length(FixedSizeWeightVector(UInt(10))) == 10
end

include("DynamicDiscreteSampler_tests.jl") # Indirect tests for an upstream usage/legacy API

@testset "Recompute shift overflow stress test" begin
    function recompute_shift_stress_test()
        w = WeightVector(2^(24-7)+1)
        w[end] = floatmax()/2^7
        w .= floatmax()
        w[2098:end] .= nextfloat(0.0)
        w[1:2097] .= ldexp.(nextfloat(0.0), 1:2097)
        w.m[5] < UInt64(2)^32 || error("The pathological constructor isn't constructing properly")
        i = 0
        while w.m[5] < UInt64(2)^32
            rand(w)
            i += 1
            i > 2^25 && error("Unexpectedly not hitting the recompute m5 edge case (p=1e-14)")
        end
        true
    end
    @test recompute_shift_stress_test()
end

# These tests are too slow:
if "CI" in keys(ENV)
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(WeightVectors, deps_compat=false)
        Aqua.test_deps_compat(WeightVectors, check_extras=false)
    end
end

include("weights.jl")
