
using AliasTables

const tables = [
    AliasTable{UInt128}(UInt128.([binomial(1,i) for i in 0:1])), 
    AliasTable{UInt128}(UInt128.([binomial(2,i) for i in 0:2])),
    AliasTable{UInt128}(UInt128.([binomial(4,i) for i in 0:4])), 
    AliasTable{UInt128}(UInt128.([binomial(8,i) for i in 0:8])),
    AliasTable{UInt128}(UInt128.([binomial(16,i) for i in 0:16])), 
    AliasTable{UInt128}(UInt128.([binomial(32,i) for i in 0:32])),
    AliasTable{UInt128}(UInt128.([binomial(64,i) for i in 0:64])), 
    AliasTable{UInt128}(UInt128.([binomial(BigInt(128),i) for i in 0:128]))
]

# implementation based on Farach-Colton, M. and Tsai, M.T., 2015. Exact sublinear binomial sampling
function binomial_int(rng, trials, px, py)
    if trials == 0 || px == 0
		return 0
    elseif px == py
        return trials
    end
    count = 0
    if px * 2 == py
        count += binomial_int_12(rng, trials)
    else
        pw = Rational(px, py)
        pt = Rational(1, 2)
        while trials > 0
            c = binomial_int_12(rng, trials)
            if pw >= pt
                count += c
                trials -= c
                pw -= pt
            else
                trials = c
            end
            pt /= 2
        end
    end
    return count
end

function binomial_int_12(rng, trials)
    count = 0
    @inbounds while trials != 0
        p = min(7, exponent(trials)) + 1
        count += rand(rng, tables[p]) - 1
        trials -= 1 << (p-1)
    end
    return count
end

function multinomial_int(rng, trials, weights)
    sum_weights = sum(weights)
    counts = Vector{Int}(undef, length(weights))
    @inbounds for i in 1:length(weights)
        b = binomial_int(rng, trials, weights[i], sum_weights)
        counts[i] = b
        trials -= b
        trials == 0 && break
        sum_weights -= weights[i]
    end
    return counts
end
