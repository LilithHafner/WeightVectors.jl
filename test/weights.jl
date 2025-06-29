using WeightVectors, Test

@test WeightVectors.FixedSizeWeightVector(10) isa WeightVectors.FixedSizeWeightVector
@test WeightVectors.WeightVector(10) isa WeightVectors.WeightVector

w = WeightVectors.FixedSizeWeightVector(10)

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

@test iszero(w) == false
for i in 1:10
    w[i] = 0
end
@test iszero(w) == true

w = WeightVectors.FixedSizeWeightVector(10)
w[1] = 3
w[2] = 2
w[3] = 3
w = FixedSizeWeightVector(w)
@test w[1] == 3
@test w[2] == 2
@test w[3] == 3

w = WeightVectors.FixedSizeWeightVector(10)
w[9] = 3
w[7] = 3
w[1] = 3
w = copy(w)
@test w[9] == 3
@test w[7] == 3
@test w[1] == 3

w = WeightVectors.FixedSizeWeightVector(10)
w[8] = 0.549326222415666
w[6] = 1.0149666786255531
w[3] = 0.8210275222825218
@test w[8] === 0.549326222415666
@test w[6] === 1.0149666786255531
@test w[3] === 0.8210275222825218

w = WeightVectors.FixedSizeWeightVector(10)
w[8] = 3.2999782300326728
w[9] = 0.7329714939310719
w[3] = 2.397108987310203
@test w[8] === 3.2999782300326728
@test w[9] === 0.7329714939310719
@test w[3] === 2.397108987310203

w = WeightVectors.FixedSizeWeightVector(10)
w[1] = 1.5
w[2] = 1.6
w[1] = 1.7
@test w[1] === 1.7
@test w[2] === 1.6

w = WeightVectors.FixedSizeWeightVector(10)
w[1] = 1
w[2] = 1e8
@test w[1] == 1
@test w[2] === 1e8

let w = WeightVectors.FixedSizeWeightVector(2)
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

let w = WeightVectors.FixedSizeWeightVector(10)
    for i in 1:40
        w[1] = 1.5*2.0^i
        @test w[1] === 1.5*2.0^i
    end
end

w = WeightVectors.WeightVector(10)
resize!(w, 20)
resize!(w, unsigned(30))

w = WeightVectors.WeightVector(10)
w[5] = 3
resize!(w, 20)
v = fill(0.0, 20)
v[5] = 3
@test w == v

@test rand(w) == 5
w[11] = v[11] = 3.5
@test w == v

w = WeightVectors.WeightVector(10)
w[1] = 1.2
w[1] = 0
resize!(w, 20)
w[15] = 1.3
@test w[11] == 0

w = WeightVectors.WeightVector(10)
w[1] = 1.2
w[2] = 1.3
w[2] = 0
resize!(w, 20)

w = WeightVectors.WeightVector(10)
w[5] = 1.2
w[6] = 1.3
w[6] = 0
resize!(w, 20)
w[15] = 2.1
resize!(w, 40)
w[30] = 4.1
w[22] = 2.2 # This previously threw

w = WeightVectors.WeightVector(10);
w[5] = 1.5
resize!(w, 3)
resize!(w, 20) # This previously threw
@test w == fill(0.0, 20)

w = WeightVectors.WeightVector(2)
w[1] = .3
w[2] = 1.1
w[2] = .4
w[2] = 2.1
w[1] = .6
w[2] = .7 # This used to throw
@test w == [.6, .7]

w = WeightVectors.WeightVector(1)
w[1] = 18
w[1] = .9
w[1] = 1.3
w[1] = .01
w[1] = .9
@test w == [.9]
resize!(w, 2)
@test w == [.9, 0]

w = WeightVectors.WeightVector(2)
w[2] = 19
w[2] = 10
w[2] = .9
w[1] = 2.1
w[1] = 1.1
w[1] = 0.7
@test w == [.7, .9]

w = WeightVectors.WeightVector(6)
resize!(w, 2108)
w[296] = 3.686559798150465e39
w[296] = 0
w[1527] = 1.0763380850925863
w[355] = 0.01640346013465141
w[881] = 79.54017710382257
w[437] = 3.848925751307115
w[571] = 1.0339246678117338
w[762] = 0.7965409844985439
w[1814] = 1.3864105787251011e-12
w[881] = 0
w[1059] = 0.9443147177405427
w[668] = 255825.83047903617
w[23] = 1.0173292806984486
w[377] = 6.652796808681465
w[668] = 0
w[1939] = 7.075668919342077e18
w[979] = 0.8922993294513122
resize!(w, 1612) # This previously threw an AssertionError: 48 <= Base.top_set_bit(m[4]) <= 49

w = WeightVectors.FixedSizeWeightVector(10)
for x in (floatmin(Float64), prevfloat(1.0, 2), prevfloat(1.0), 1.0, nextfloat(1.0), nextfloat(1.0, 2), floatmax(Float64))
    w[1] = x # This previously threw on prevfloat(1.0) and floatmax(Float64)
    @test w[1] === x
end

include("invariants.jl")

w = WeightVectors.WeightVector(31)
w[11] = 9.923269000574892e-8
w[23] = 0.9876032886161744
w[31] = 1.1160998022859043
verify(w.m)

w = WeightVectors.FixedSizeWeightVector(10)
w[1] = floatmin(Float64)
w[2] = floatmax(Float64)
w[2] = 0 # This previously threw an assertion error due to overflow when estimating sum of level weights
verify(w.m)
w[1] = eps(0.0)
@test w[1] == eps(0.0)
verify(w.m)

# Confirm that FixedSizeWeightVector cannot be resized:
w = WeightVectors.FixedSizeWeightVector(10)
@test_throws MethodError resize!(w, 20)
@test_throws MethodError resize!(w, 5)
w2 = WeightVectors.WeightVector(w)
resize!(w2, 5)
@test length(w2) == 5
@test length(w) == 10 # The fixed size has not changed

w = WeightVectors.FixedSizeWeightVector(9)
v = zeros(9)
v[4] = w[4] = 2.44
v[5] = w[5] = 0.76
v[6] = w[6] = 0.61
v[7] = w[7] = 0.62
v[9] = w[9] = 2.15
v[1] = w[1] = 1.65
v[7] = w[7] = 1.46
v[8] = w[8] = 0.25
v[2] = w[2] = 0.93
v[3] = w[3] = 3.67
v[6] = w[6] = 9.92
v[5] = w[5] = 1.72
v[6] = w[6] = 0.70
v[8] = w[8] = 0.72
v[5] = w[5] = 0.20
v[1] = w[1] = 0.71
v[3] = w[3] = 0.92
verify(w.m)
@test v == w

w = WeightVectors.WeightVector(2)
w[1] = 0.95
w[2] = 6.41e14
verify(w.m)

# This test catches a bug that was not revealed by the RNG tests below
w = WeightVectors.FixedSizeWeightVector(3);
w[1] = 1.5
w[2] = prevfloat(1.5)
w[3] = 2^25
verify(w.m)

# This test catches a bug that was not revealed by the RNG tests below.
# The final line is calibrated to have about a 50% fail rate on that bug
# and run in about 3 seconds:
w = WeightVectors.FixedSizeWeightVector(2046*2048)
w .= repeat(ldexp.(1.0, -1022:1023), inner=2048)
w[(2046-16)*2048+1:2046*2048] .= 0
@test w.m[4] < 2.0^32*1.1 # Confirm that we created an interesting condition
f(w,n) = sum(Int64(rand(w)) for _ in 1:n)
verify(w.m)
@test f(w, 2^27) ≈ 4.1543685e6*2^27 rtol=1e-6 # This should fail less than 1e-40 of the time

# These tests have never revealed a bug that was not revealed by one of the above tests:
w = WeightVectors.FixedSizeWeightVector(10)
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
        w = WeightVectors.FixedSizeWeightVector(10)
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

# This alone probably catches all bugs that are caught by tests above (with one exception).
# However, whenever we identify and fix a bug, we add a specific test for it above.
include("statistical.jl")
try
    let
        print("weights.jl randomized tests: 0%")
        for rep in 1:1000
            if rep % 10 == 0
                print("\rweights.jl randomized tests: $(rep÷10)%")
            end
            global LOG = []
            len = rand(1:100)
            push!(LOG, len)
            w = WeightVectors.WeightVector(len)
            v = fill(0.0, len)
            resize = rand(Bool) # Some behavior emerges when not resizing for a long period
            opcount = 0
            m10531 = 0
            function track_and_test_compaction_frequency(w)
                opcount += 1
                if w.m[10531] < m10531 # compaction has occurred
                    @assert opcount > length(w)/8
                    opcount = 0
                end
                m10531 = w.m[10531]
            end
            for _ in 1:rand((10,100,3000))
                @test v == w
                verify(w.m)
                if rand() < .01
                    sm = sum(big, v)
                    sm == 0 || statistical_test(w, Float64.(v ./ sm))
                end
                x = rand()
                if x < .2 && !all(iszero, v)
                    i = rand(findall(!iszero, v))
                    push!(LOG, i => 0)
                    v[i] = 0
                    w[i] = 0
                    track_and_test_compaction_frequency(w)
                elseif x < .4 && !all(iszero, v)
                    i = rand(w)
                    push!(LOG, i => 0)
                    v[i] = 0
                    w[i] = 0
                    track_and_test_compaction_frequency(w)
                elseif x < .9 || !resize
                    i = rand(eachindex(v))
                    x = if x < .41
                        reinterpret(Float64, rand(0:reinterpret(UInt64, floatmin(Float64))))
                    else
                        min(exp(rand((.1, 7, 300))*randn()), floatmax(Float64))
                    end
                    push!(LOG, i => x)
                    v[i] = x
                    w[i] = x
                    track_and_test_compaction_frequency(w)
                else
                    l_old = length(v)
                    l_new = rand(1:rand((10,100,3000)))
                    push!(LOG, resize! => l_new)
                    resize!(v, l_new)
                    resize!(w, l_new)
                    m10531 = w.m[10531]
                    if l_new > l_old
                        v[l_old+1:l_new] .= 0
                    end
                end
            end
        end
        println()
    end
    println("These tests should fail due to random noise no more than $FALSE_POSITIVITY_ACCUMULATOR of the time")
catch
    println("Reproducer:\n```julia")
    for L in LOG
        if L isa Int
            println("w = WeightVectors.WeightVector($L)")
        elseif first(L) === resize!
            println("resize!(w, $(last(L)))")
        else
            println("w[$(first(L))] = $(last(L))")
        end
    end
    println("```")
    rethrow()
end
