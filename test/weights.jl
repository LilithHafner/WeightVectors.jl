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
@test w[7] == 3
@test w[1] == 3

w = DynamicDiscreteSamplers.FixedSizeWeights(10)
w[8] = 0.549326222415666
w[6] = 1.0149666786255531
w[3] = 0.8210275222825218
@test w[8] === 0.549326222415666
@test w[6] === 1.0149666786255531
@test w[3] === 0.8210275222825218

w = DynamicDiscreteSamplers.FixedSizeWeights(10)
w[8] = 3.2999782300326728
w[9] = 0.7329714939310719
w[3] = 2.397108987310203
@test w[8] === 3.2999782300326728
@test w[9] === 0.7329714939310719
@test w[3] === 2.397108987310203

w = DynamicDiscreteSamplers.FixedSizeWeights(10)
w[1] = 1.5
w[2] = 1.6
w[1] = 1.7
@test w[1] === 1.7
@test w[2] === 1.6

w = DynamicDiscreteSamplers.FixedSizeWeights(10)
w[1] = 1
w[2] = 1e8
@test w[1] == 1
@test w[2] === 1e8

let w = DynamicDiscreteSamplers.FixedSizeWeights(2)
    w[1] = 1.1
    w[2] = 1.9
    twos = 0
    n = 10_000
    for _ in 1:n
        x = rand(w)
        @test x ∈ 1:2
        if x == 2
            twos += 1
        end
    end
    @test (w[2]/(w[1]+w[2])) === 1.9/3
    expected = n*1.9/3
    stdev = .5sqrt(n)
    @test abs(twos-expected) < 4stdev
end

let w = DynamicDiscreteSamplers.FixedSizeWeights(10)
    for i in 1:40
        w[1] = 1.5*2.0^i
        @test w[1] === 1.5*2.0^i
    end
end

w = DynamicDiscreteSamplers.ResizableWeights(10)
resize!(w, 20)
resize!(w, unsigned(30))

w = DynamicDiscreteSamplers.ResizableWeights(10)
w[5] = 3
resize!(w, 20)
v = fill(0.0, 20)
v[5] = 3
@test w == v

@test rand(w) == 5
w[11] = v[11] = 3.5
@test w == v

# These tests have never revealed a bug that was not revealed by one of the above tests:
w = DynamicDiscreteSamplers.FixedSizeWeights(10)
w[1] = 1
w[2] = 1e100
@test rand(w) === 2
w[3] = 1e-100
@test rand(w) === 2
w[2] = 0
@test rand(w) === 1
w[1] = 0
@test rand(w) === 3
w[3] = 0
@test_throws ArgumentError("collection must be non-empty") rand(w)

let
    for _ in 1:10000
        w = DynamicDiscreteSamplers.FixedSizeWeights(10)
        v = [w[i] for i in 1:10]
        for _ in 1:10
            i = rand(1:10)
            x = rand((0.0, exp(10randn())))
            w[i] = x
            v[i] = x
            @test all(v[i] === w[i] for i in 1:10)
        end
    end
end
