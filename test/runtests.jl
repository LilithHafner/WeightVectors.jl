using DynamicDiscreteSamplers
using Test
using Aqua

@testset "DynamicDiscreteSamplers.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DynamicDiscreteSamplers, deps_compat=false)
        Aqua.test_deps_compat(DynamicDiscreteSamplers, check_extras=false)
    end
    # Write your tests here.
end
