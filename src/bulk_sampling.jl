
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
