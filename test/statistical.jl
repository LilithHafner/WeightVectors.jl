using HypothesisTests, Random

FALSE_POSITIVITY_ACCUMULATOR::Float64 = isdefined(@__MODULE__, :FALSE_POSITIVITY_ACCUMULATOR) ? FALSE_POSITIVITY_ACCUMULATOR : 0.0;

function _statistical_test(rng, sampler, expected_probabilities, n, p)
    sample = similar(expected_probabilities, Int)
    sample .= 0
    for _ in 1:n
        sample[rand(rng, sampler)] += 1
    end
    chisq_test = ChisqTest(sample, expected_probabilities)
    pvalue(chisq_test) > p
end

function _statistical_test(rng, sampler, expected_probabilities)
    for i in 0:5
        n = 1_000 * 10^i
        p = .1 * .1^i
        _statistical_test(rng, sampler, expected_probabilities, n, p) && return true
    end
    println(stderr, "statistical test failure")
    global FAILED_SAMPLER = sampler
    global FAILED_EXPECTED_PROBABILITIES = expected_probabilities
    if isinteractive()
        println("reproduce with statistical_test(FAILED_SAMPLER, FAILED_EXPECTED_PROBABILITIES)")
    else
        @show sampler expected_probabilities
        @show sampler.m
    end
    false
end

function statistical_test(rng, sampler, expected_probabilities)
    global FALSE_POSITIVITY_ACCUMULATOR += .1*.1^7
    @test _statistical_test(rng, sampler, expected_probabilities)
end
statistical_test(sampler, expected_probabilities) =
    statistical_test(Random.default_rng(), sampler, expected_probabilities)
