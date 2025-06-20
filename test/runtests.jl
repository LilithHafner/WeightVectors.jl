using WeightVectors
using HypothesisTests
using Random
using Test
using Aqua
using StableRNGs
using StatsBase

@test WeightVectors.DEBUG === true

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

# Test effects
# TODO: make effects good even with good error messages and precompilation
effects_code = String(read(pathof(WeightVectors)))
src = dirname(pathof(WeightVectors))
while true # Inclusion, assuming no nested directories
    m = match(r"(include\(\"(.*?)\"\))", effects_code)
    m === nothing && break
    global effects_code = replace(effects_code, m.captures[1]=>"begin\n$(String(read(joinpath(src, m.captures[2]))))\nend")
end

effects_code = replace(effects_code, "@assert"=>"#@assert") # Asserts have bad effects
effects_code = replace(effects_code, "\nprecompile("=>"\n#precompile(") # Precompile hurts effects (https://github.com/JuliaLang/julia/issues/57324)
effects_code = replace(effects_code, r"throw\((Bounds|Argument|Domain)Error\(.*?\)\)"=>"error()") # Good errors have bad effects
if VERSION >= v"1.11"
    effects_code = replace(effects_code, r"\n.* copyto!\(.*\n" => "\n") # https://github.com/JuliaLang/julia/issues/58750
end
effects_file = tempname()
open(effects_file, "w") do io
    write(io, effects_code)
end
module EffectsWorkaround
    include(parentmodule(@__MODULE__).effects_file)
end
@testset "Effects" begin
    WV = EffectsWorkaround.WeightVectors
    TRUE = Core.Compiler.ALWAYS_TRUE
    for T in [WV.WeightVector, WV.FixedSizeWeightVector]
        e = Base.infer_effects(rand, (Xoshiro, T))
        @test e.consistent != TRUE
        @test e.effect_free == Core.Compiler.EFFECT_FREE_IF_INACCESSIBLEMEMONLY
        @test e.nothrow == false # in the case of a malformed sampler
        @test e.terminates == false # it's plausible this could not terminate for pathological RNG state (e.g. all zeros)
        @test e.notaskstate
        @test e.inaccessiblememonly == Core.Compiler.INACCESSIBLEMEM_OR_ARGMEMONLY
        VERSION >= v"1.11" && @test e.noub == TRUE
        VERSION >= v"1.11" && @test e.nonoverlayed == TRUE
        VERSION >= v"1.11" && @test e.nortcall

        e = Base.infer_effects(getindex, (T, Int))
        @test e.consistent != TRUE
        @test e.effect_free == TRUE
        @test e.nothrow == false # index out of bounds
        @test e.terminates
        @test e.notaskstate
        @test e.inaccessiblememonly == Core.Compiler.INACCESSIBLEMEM_OR_ARGMEMONLY
        VERSION >= v"1.11" && @test e.noub == TRUE
        VERSION >= v"1.11" && @test e.nonoverlayed == TRUE
        VERSION >= v"1.11" && @test e.nortcall

        e = Base.infer_effects(setindex!, (T, Float64, Int))
        @test e.consistent != TRUE
        VERSION >= v"1.12" && @test e.effect_free == Core.Compiler.EFFECT_FREE_IF_INACCESSIBLEMEMONLY # broken due to copyto!(::Memory, ::Int, ::Memory, ::Int, ::Int), which is hacked out in 1.12+
        @test e.nothrow == false # index out of bounds
        @test_broken e.terminates # loop analysis is weak
        VERSION >= v"1.11" && @test e.notaskstate
        VERSION >= v"1.12" && @test e.inaccessiblememonly == Core.Compiler.INACCESSIBLEMEM_OR_ARGMEMONLY # broken due to copyto!(::Memory, ::Int, ::Memory, ::Int, ::Int), which is hacked out in 1.12+
        VERSION >= v"1.11" && @test e.noub == TRUE
        VERSION >= v"1.11" && @test e.nonoverlayed == TRUE
        VERSION >= v"1.11" && @test e.nortcall
    end

    for T in [WV.WeightVector]
        e = Base.infer_effects(resize!, (T, Int))
        @test e.consistent != TRUE
        VERSION >= v"1.12" && @test e.effect_free == Core.Compiler.EFFECT_FREE_IF_INACCESSIBLEMEMONLY # broken due to copyto!(::Memory, ::Int, ::Memory, ::Int, ::Int), which is hacked out in 1.12+
        @test e.nothrow == false # index out of bounds
        @test_broken e.terminates # loop analysis is weak
        VERSION >= v"1.11" && @test e.notaskstate
        VERSION >= v"1.12" && @test e.inaccessiblememonly == Core.Compiler.INACCESSIBLEMEM_OR_ARGMEMONLY # broken due to copyto!(::Memory, ::Int, ::Memory, ::Int, ::Int), which is hacked out in 1.12+
        VERSION >= v"1.11" && @test e.noub == TRUE
        VERSION >= v"1.11" && @test e.nonoverlayed == TRUE
        VERSION >= v"1.11" && @test e.nortcall
    end
end

# These tests are too slow:
if "CI" in keys(ENV)
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(WeightVectors, deps_compat=false)
        Aqua.test_deps_compat(WeightVectors, check_extras=false)
    end
end

include("weights.jl")
