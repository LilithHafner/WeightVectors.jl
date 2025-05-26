using DynamicDiscreteSamplers
using HypothesisTests
using Random
using Test
using Aqua
using StableRNGs
using StatsBase

@test DynamicDiscreteSamplers.DEBUG === true

# These tests are too slow:
if "CI" in keys(ENV)
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DynamicDiscreteSamplers, deps_compat=false)
        Aqua.test_deps_compat(DynamicDiscreteSamplers, check_extras=false)
    end
end

include("weights.jl")
