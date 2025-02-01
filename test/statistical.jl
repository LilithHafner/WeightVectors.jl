using StatsFuns, Random

# Not using HypothesisTests's ChisqTest because of https://github.com/JuliaStats/HypothesisTests.jl/issues/281
# Not using ChisqTest at all because it reports very low p values with high probability for some distributions

"""
    statistical_test(rng, sampler, expected_probabilities, n)

For all `p`, if `sampler` samples according to `expected_probabilities` then return a number
less than `p` with probability at most `p`.

Also, makes an effort to return low numbers whenever `sampler` does not sample according to
`expected_probabilities`.

Calls `rand(rng, sampler)` `n` times.

Inspired by https://github.com/JuliaStats/Distributions.jl/blob/1e6801da6678164b13330cc1f16e670768d27330/test/testutils.jl#L99
"""
function _statistical_test(rng, sampler, expected_probabilities, n)
    sample = similar(expected_probabilities, Int)
    sample .= 0
    for _ in 1:n
        sample[rand(rng, sampler)] += 1
    end

    nonzeros = 0
    for i in eachindex(expected_probabilities, sample)
        if iszero(expected_probabilities[i])
            sample[i] != 0 && return 0.0
        else
            nonzeros += 1
        end
    end

    mn = 1/2nonzeros # TODO: I think this 2 can go away.
    for i in eachindex(expected_probabilities, sample)
        if !iszero(expected_probabilities[i])
            p_le = binomcdf(n, expected_probabilities[i], sample[i])
            p_ge = 1-binomcdf(n, expected_probabilities[i], sample[i]-1)
            mn = min(mn, p_le, p_ge)
        end
    end
    mn*2nonzeros
end

FALSE_POSITIVITY_ACCUMULATOR::Float64 = isdefined(@__MODULE__, :FALSE_POSITIVITY_ACCUMULATOR) ? FALSE_POSITIVITY_ACCUMULATOR : 0.0;

function _statistical_test(rng, sampler, expected_probabilities)
    global FALSE_POSITIVITY_ACCUMULATOR += 1e-8

    p = _statistical_test(rng, sampler, expected_probabilities, 1_000)
    p > .1 && return true
    for _ in 1:7
        p = _statistical_test(rng, sampler, expected_probabilities, 10_000)
        p > .1 && return true
    end

    println(stderr, "statistical test failure")
    global FAILED_SAMPLER = sampler
    global FAILED_EXPECTED_PROBABILITIES = expected_probabilities
    if isinteractive()
        println("reproduce with `statistical_test(FAILED_SAMPLER, FAILED_EXPECTED_PROBABILITIES)`")
    else
        @show sampler expected_probabilities
        @show sampler.m
    end
    false
end

function statistical_test(rng, sampler, expected_probabilities)
    @test _statistical_test(rng, sampler, expected_probabilities)
end
statistical_test(sampler, expected_probabilities) =
    statistical_test(Random.default_rng(), sampler, expected_probabilities)
