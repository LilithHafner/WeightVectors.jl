using DynamicDiscreteSamplers, Test

@test DynamicDiscreteSamplers.FixedSizeWeights(10) isa DynamicDiscreteSamplers.FixedSizeWeights
@test DynamicDiscreteSamplers.ResizableWeights(10) isa DynamicDiscreteSamplers.ResizableWeights
@test DynamicDiscreteSamplers.SemiResizableWeights(10) isa DynamicDiscreteSamplers.SemiResizableWeights

w = DynamicDiscreteSamplers.FixedSizeWeights(10)

@test_throws ArgumentError("collection must be non-empty") rand(w)

@test 1 === (w[1] = 1)

@test rand(w) === 1

@test_throws BoundsError w[0]
@test_throws BoundsError w[11]
@test w[1] === 1.0
for i in 2:10
    @test w[i] === 0.0
end

@test 0 === (w[1] = 0)
@test w[1] === 0.0

@test_throws ArgumentError("collection must be non-empty") rand(w)

w[1] = 1.5
@test w[1] === 1.5

w[1] = 2
@test w[1] === 2.0

w = DynamicDiscreteSamplers.FixedSizeWeights(10)
w[1] = 3
w[2] = 2
w[3] = 3
@test w[1] == 3
@test w[2] == 2
@test w[3] == 3

w = DynamicDiscreteSamplers.FixedSizeWeights(10)
w[9] = 3
w[7] = 3
w[1] = 3
@test w[9] == 3
@test_broken w[7] == 3
@test w[1] == 3

# These tests have never revealed a bug:
for _ in 1:100
    local w = DynamicDiscreteSamplers.FixedSizeWeights(10)
    local v = [w[i] for i in 1:10]
    for _ in 1:2
        i = rand(1:10)
        x = exp(randn())
        w[i] = x
        v[i] = x
        @test all(v[i] === w[i] for i in 1:10)
    end
end
