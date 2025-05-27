
using AliasTables

# These are exact because their weights sum to powers of 2.
const ALIASTABLES = (
    AliasTable{UInt64}(UInt64.([binomial(BigInt(1),i) for i in 0:1])), 
    AliasTable{UInt64}(UInt64.([binomial(BigInt(2),i) for i in 0:2])),
    AliasTable{UInt64}(UInt64.([binomial(BigInt(4),i) for i in 0:4])), 
    AliasTable{UInt64}(UInt64.([binomial(BigInt(8),i) for i in 0:8])),
    AliasTable{UInt64}(UInt64.([binomial(BigInt(16),i) for i in 0:16])), 
    AliasTable{UInt64}(UInt64.([binomial(BigInt(32),i) for i in 0:32])),
    AliasTable{UInt64}(UInt64.([binomial(BigInt(64),i) for i in 0:64]))
)

function _rand(rng::AbstractRNG, m::Memory{UInt64}, n::Integer)
    n < 100 && return [_rand(rng, m) for _ in 1:n]
    max_i = _convert(Int, m[2])
    min_i = 5
    k = 0
    @inbounds for j in 10235:10266
        chunk = m[j]
        if k == 0 && chunk != 0
            min_i = (j-10235) << 6 + leading_zeros(chunk) + 4
        end
        k += count_ones(chunk)
    end
    n < 100*(k^0.72) && return [_rand(rng, m) for _ in 1:n]
    inds = Vector{Int}(undef, k)
    weights = Vector{BigInt}(undef, k)
    q = 0
    @inbounds for j in max_i:-1:min_i
        if m[j] != 0
            q += 1
            inds[q] = j
            weights[q] = BigInt(get_significand_sum(m, j)) << (j-min_i)
        end
    end
    counts = multinomial_sample(rng, n, weights)
    samples = Vector{Int}(undef, n)
    ct, s = 0, 1
    @inbounds for i in 1:k
        c = counts[i]
        c == 0 && continue
        j = 2*inds[i] + 6133
        pos = m[j]
        len = m[j+1]
        for _ in 1:c
            samples[s] = sample_within_level(rng, m, pos, len)
            s += 1
        end
        ct += c
        ct == n && break
    end
    return faster_shuffle!(rng, samples)
end

# it uses some internals from Base.GMP.MPZ (as MutableArithmetics.jl) to speed-up some BigInt operations
"""
    binomial_sample(rng, trials, px, py)
    
Flip a coin with probability of `px//py` of coming up heads `trials` times and return the number of heads.

Has `O(trials)` expected runtime with a very low constant factor.

Implementation based on Farach-Colton, M. and Tsai, M.T., 2015. Exact sublinear binomial sampling.
"""
function binomial_sample(rng, trials, px, py)
    if iszero(trials) || iszero(px)
        return 0
    elseif px == py
        return trials
    end
    count = 0
    while trials > 0
        c = binomial_sample_fair_coin(rng, trials)
        Base.GMP.MPZ.mul_2exp!(px, 1) # px *= 2
        if px > py
            count += c
            trials -= c
            Base.GMP.MPZ.sub!(px, py) # px -= py
        elseif px < py
            trials = c
        else
            count += c
            break
        end
    end
    return count
end

"""
    binomial_sample_fair_coin(rng, trials)
    
Flips `trials` fair coins and reports the number of heads.

Flips up to 64 coins at a time.
"""
function binomial_sample_fair_coin(rng, trials)
    count = 0
    @inbounds while trials != 0
        p = min(6, exponent(trials))
        n = trials >> p
        table = ALIASTABLES[p+1]
        for _ in 1:n
            count += rand(rng, table)
        end
        count -= n
        trials -= n * (1 << p)
    end
    return count
end

"""
    multinomial_sample(rng, trials, weights)
    
Draw `trials` elements from the probability distribution specified by `weights` (need not sum to 1) and return the number of times each element was drawn.

Runs in `O(trials * weights)`, but can be as fast as `O(trials)` if the weights are skewed toward big weights at the beginning.
"""
function multinomial_sample(rng, trials, weights::AbstractVector{<:Integer})
    sum_weights = sum(weights)
    counts = Vector{Int}(undef, length(weights))
    weight_copy = BigInt(0)
    @inbounds for i in 1:length(weights)-1
        weight = weights[i]
        Base.GMP.MPZ.set!(weight_copy, weight)
        b = binomial_sample(rng, trials, weight_copy, sum_weights)
        counts[i] = b
        trials -= b
        trials == 0 && return counts
        Base.GMP.MPZ.sub!(sum_weights, weight)
    end
    counts[end] = trials
    return counts
end

function faster_shuffle!(rng::AbstractRNG, vec::AbstractVector)
    for i in 2:length(vec)
        endi = (i-1) % UInt
        j = @inline rand(rng, Random.Sampler(rng, UInt(0):endi, Val(1))) % Int + 1
        vec[i], vec[j] = vec[j], vec[i]
    end
    vec
end
