using DynamicDiscreteSamplers
using Random
using Test
using Aqua

@testset "basic end to end tests" begin
    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1.0)
    push!(ds, 2, 2.0)
    push!(ds, 3, 4.0)
    delete!(ds, 1)
    delete!(ds, 2)
    @test rand(ds) == 3
    push!(ds, 1, 3.0)
    delete!(ds, 1)
    @test rand(ds) == 3

    ds = DynamicDiscreteSampler()
    push!(ds, 1, 5.0)
    push!(ds, 2, 6.0)
    delete!(ds, 1)
    delete!(ds, 2)
end

@testset "randomized end to end tests" begin
    ds = DynamicDiscreteSampler()
    x = randperm(100)
    y = exp.(10*rand(100).-5);
    push!.((ds,), x, y)
    for _ in 1:100
        rand(ds)
    end
    # for i in randperm(99)
    #     delete!(ds, i)
    # end
    # @test rand(ds) == 100
end

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(DynamicDiscreteSamplers, deps_compat=false, persistent_tasks=false)
    Aqua.test_deps_compat(DynamicDiscreteSamplers, check_extras=false)
end
