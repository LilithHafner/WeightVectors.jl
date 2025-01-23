using DynamicDiscreteSamplers, Test

@test DynamicDiscreteSamplers.FixedSizeWeights(10) isa DynamicDiscreteSamplers.FixedSizeWeights
@test DynamicDiscreteSamplers.ResizableWeights(10) isa DynamicDiscreteSamplers.ResizableWeights
@test DynamicDiscreteSamplers.SemiResizableWeights(10) isa DynamicDiscreteSamplers.SemiResizableWeights

w = DynamicDiscreteSamplers.FixedSizeWeights(10)

@test_throws ArgumentError("collection must be non-empty") rand(w)

@test 1 === (w[1] = 1)

@test_broken rand(w) === 1
