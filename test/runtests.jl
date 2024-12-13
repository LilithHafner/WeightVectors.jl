using DynamicDiscreteSamplers
using Random
using Test
using Aqua

@testset "unit tests" begin
    lls = DynamicDiscreteSamplers.LinkedListSet3()
    push!(lls, 2)
    push!(lls, 3)
    delete!(lls, 2)
    @test 3 in lls
end

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

    ds = DynamicDiscreteSampler()
    push!(ds, 1, 1.0)
    push!(ds, 2, 2.0)
    delete!(ds, 2)

    ds = DynamicDiscreteSampler()
    for i in 1:65
        push!(ds, i, 2.0^i)
    end
    delete!(ds, 65)
    push!(ds, 65, 1.0)
    delete!(ds, 64)
end

@testset "randomized end to end tests" begin
    ds = DynamicDiscreteSampler()
    x = randperm(100)
    y = exp.(10*rand(100).-5);
    push!.((ds,), x, y)
    for _ in 1:100
        rand(ds)
    end
    for i in randperm(99)
        delete!(ds, i)
    end
    @test rand(ds) == 100
end

# @testset "interleaved randomized end to end tests" begin
#     Random.seed!()
#     ds = DynamicDiscreteSampler()
#     elements = Set{Int}()
#     for i in 1:30000
#         if rand() < 0.5
#             i = rand(1:10000)
#             if i ∉ elements
#                 push!(ds, i, exp(100randn()))
#                 push!(elements, i)
#             end
#         elseif length(elements) > 0
#             element = rand(elements)
#             delete!(ds, element)
#             delete!(elements, element)
#         end

#         if length(elements) > 0
#             @test rand(ds) in elements
#         end
#     end
# end

# These tests are too slow:
# @testset "Code quality (Aqua.jl)" begin
#     Aqua.test_all(DynamicDiscreteSamplers, deps_compat=false, persistent_tasks=false)
#     Aqua.test_deps_compat(DynamicDiscreteSamplers, check_extras=false)
# end
