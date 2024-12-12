using DynamicDiscreteSampler
using Test
using Aqua

@testset "DynamicDiscreteSampler.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DynamicDiscreteSampler)
    end
    # Write your tests here.
end
