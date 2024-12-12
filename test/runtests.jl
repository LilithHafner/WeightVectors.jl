using DynamicDiscreteSampler
using Test
using Aqua

@testset "DynamicDiscreteSampler.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DynamicDiscreteSampler, deps_compat=false)
        Aqua.test_deps_compat(DynamicDiscreteSampler, check_extras=false)
    end
    # Write your tests here.
end
