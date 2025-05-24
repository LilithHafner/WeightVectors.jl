
using AliasTables

const ALIASTABLES = (
    AliasTable{UInt128}(UInt128.([binomial(BigInt(1),i) for i in 0:1])), 
    AliasTable{UInt128}(UInt128.([binomial(BigInt(2),i) for i in 0:2])),
    AliasTable{UInt128}(UInt128.([binomial(BigInt(4),i) for i in 0:4])), 
    AliasTable{UInt128}(UInt128.([binomial(BigInt(8),i) for i in 0:8])),
    AliasTable{UInt128}(UInt128.([binomial(BigInt(16),i) for i in 0:16])), 
    AliasTable{UInt128}(UInt128.([binomial(BigInt(32),i) for i in 0:32])),
    AliasTable{UInt128}(UInt128.([binomial(BigInt(64),i) for i in 0:64])), 
    AliasTable{UInt128}(UInt128.([binomial(BigInt(128),i) for i in 0:128]))
)

# implementation based on Farach-Colton, M. and Tsai, M.T., 2015. Exact sublinear binomial sampling
# it uses some internals from Base.GMP.MPZ (as MutableArithmetics.jl) to speed-up some BigInt
# operations
function binomial_int(rng, trials, px, py)
    if iszero(trials) || iszero(px)
        return 0
    elseif px == py
        return trials
    end
    count = 0
    while trials > 0
        c = binomial_int_12(rng, trials)
        Base.GMP.MPZ.mul_2exp!(px, 1) # px = px * 2^1
        if px > py
            count += c
            trials -= c
            Base.GMP.MPZ.sub!(px, py) # px = px - py
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
    binomial_int_12(rng, trials)
    
Flips `trials` fair coins and reports the number of heads.

Flips up to 128 coins at a time.
"""
function binomial_int_12(rng, trials)
    count = 0
    @inbounds while trials != 0
        p = min(7, exponent(trials))
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

function multinomial_int(rng, trials, weights)
    sum_weights = sum(weights)
    counts = Vector{Int}(undef, length(weights))
    weight_copy = BigInt(0)
    @inbounds for i in 1:length(weights)-1
        weight = weights[i]
        Base.GMP.MPZ.set!(weight_copy, weight)
        b = binomial_int(rng, trials, weight_copy, sum_weights)
        counts[i] = b
        trials -= b
        trials == 0 && return counts
        Base.GMP.MPZ.sub!(sum_weights, weight)
    end
    counts[end] = trials
    return counts
end
