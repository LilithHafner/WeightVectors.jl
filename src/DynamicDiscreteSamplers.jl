module DynamicDiscreteSamplers

export DynamicDiscreteSampler, SamplerIndices

using Distributions, Random, StaticArrays

const UPPER_LIMIT = Int64(10)^12
const MAX_CUT = typemax(UInt64)-UPPER_LIMIT+1
const RANDF = 2^11/MAX_CUT

@inline sig(x::Float64) = (reinterpret(UInt64, x) & Base.significand_mask(Float64)) + Base.significand_mask(Float64) + 1

@inline function flot(sg::UInt128, level::Integer)
    shift = Int64(8 * sizeof(sg) - 53 - leading_zeros(sg))
    x = (sg >>= shift) % UInt64
    exp = level + shift + 1022
    reinterpret(Float64, x + (exp << 52))
end

struct SelectionSampler
    p::MVector{64, Float64}
    o::MVector{64, Int16}
end
function Base.rand(rng::AbstractRNG, ss::SelectionSampler, v::Float64, lastfull::Int)
    u = v*ss.p[lastfull]
    @inbounds for i in lastfull-1:-1:1
        ss.p[i] < u && return i+1
    end
    return 1
end
function set_cum_weights!(ss::SelectionSampler, ns, reorder)
    p, lastfull = ns.sampled_level_weights, ns.track_info.lastfull
    if reorder
        ns.track_info.reset_order = 0
        if !ns.track_info.reset_distribution && issorted(@view(p[1:lastfull]))
            return ss
        end
        @inline reorder_levels(ns, ss, p, lastfull)
        ns.track_info.firstchanged = 1
    end
    firstc = ns.track_info.firstchanged
    ss.p[1] = p[1]
    f = firstc + Int(firstc == 1)
    @inbounds for i in f:lastfull
        ss.p[i] = ss.p[i-1] + p[i]
    end
    ns.track_info.firstchanged = lastfull
    return ss
end
function reorder_levels(ns, ss, p, lastfull)
    sortperm!(@view(ss.o[1:lastfull]), @view(p[1:lastfull]); alg=Base.Sort.InsertionSortAlg())
    @inbounds for i in 1:lastfull
        if ss.o[i] == zero(Int16)
            all_index = ns.level_set_map.indices[ns.sampled_level_numbers[i]+1075][1]
            ns.level_set_map.indices[ns.sampled_level_numbers[i]+1075] = (all_index, i)
            continue
        end
        value1 = ns.sampled_levels[i]
        value2 = ns.sampled_level_numbers[i]
        value3 = p[i]
        x, y = i, Int(ss.o[i])
        while y != i
            ss.o[x] = zero(Int16)
            ns.sampled_levels[x] = ns.sampled_levels[y]
            ns.sampled_level_numbers[x] = ns.sampled_level_numbers[y]
            p[x] = p[y]
            x = y
            y = Int(ss.o[x])
        end
        ns.sampled_levels[x] = value1
        ns.sampled_level_numbers[x] = value2
        p[x] = value3
        ss.o[x] = zero(Int16)
        all_index = ns.level_set_map.indices[ns.sampled_level_numbers[i]+1075][1]
        ns.level_set_map.indices[ns.sampled_level_numbers[i]+1075] = (all_index, i)
    end
end

mutable struct RejectionInfo
    length::Int
    maxw::Float64
    mask::UInt64
end
struct RejectionSampler
    data::Vector{Tuple{Int, Float64}}
    level::Int
    track_info::RejectionInfo
    RejectionSampler(level, i, v) = new([(i, v)], level, RejectionInfo(1, v, zero(UInt)))
end
function Random.rand(rng::AbstractRNG, rs::RejectionSampler, f::Function)
    len = rs.track_info.length
    mask = rs.track_info.mask
    maxw = rs.track_info.maxw
    while true
        u = rand(rng, UInt64)
        i = Int(u & mask)
        i >= len && continue
        i += 1
        res, x = rs.data[i]
        f(rng, u) * maxw < x && return (i, res)
    end
end
@inline randreuse(rng, u) = Float64(u >>> 11) * 0x1.0p-53
@inline randnoreuse(rng, _) = rand(rng)
function Base.push!(rs::RejectionSampler, i, x)
    len = rs.track_info.length += 1
    if len > length(rs.data)
        resize!(rs.data, 2*length(rs.data))
        rs.track_info.mask = UInt(1) << (8*sizeof(len-1) - leading_zeros(len-1)) - 1
    end
    rs.data[len] = (i, x)
    maxwn = rs.track_info.maxw
    rs.track_info.maxw = ifelse(x > maxwn, x, maxwn)
    rs
end
Base.isempty(rs::RejectionSampler) = length(rs) == 0 # For testing only
Base.length(rs::RejectionSampler) = rs.track_info.length # For testing only

struct LinkedListSet
    data::MVector{34, UInt64}
    LinkedListSet() = new(zero(MVector{34, UInt64}))
end
Base.in(i::Int, x::LinkedListSet) = x.data[i >> 6 + 18] & (UInt64(1) << (0x3f - (i & 0x3f))) != 0
Base.push!(x::LinkedListSet, i::Int) = (x.data[i >> 6 + 18] |= UInt64(1) << (0x3f - (i & 0x3f)); x)
Base.delete!(x::LinkedListSet, i::Int) = (x.data[i >> 6 + 18] &= ~(UInt64(1) << (0x3f - (i & 0x3f))); x)
function Base.findnext(x::LinkedListSet, i::Int)
    j = i >> 6 + 18
    k = i & 0x3f
    y = x.data[j] << k
    y != 0 && return i + leading_zeros(y)
    j2 = findnext(!iszero, x.data, j+1)
    isnothing(j2) && return nothing
    j2 << 6 + leading_zeros(x.data[j2]) - 18*64
end
function Base.findprev(x::LinkedListSet, i::Int)
    j = i >> 6 + 18
    k = i & 0x3f
    y = x.data[j] >> (0x3f - k)
    y != 0 && return i - trailing_zeros(y)
    j2 = findprev(!iszero, x.data, j-1)
    isnothing(j2) && return nothing
    j2 << 6 - trailing_zeros(x.data[j2]) - 17*64 - 1
end

# ------------------------------

#=
Each entry is assigned a level based on its power.
We have at most min(n, 2048) levels.
# Maintain a distribution over the top 64 levels and ignore any lower
(or maybe the top log(n) levels and treat the rest as a single level).
For each level, maintain a distribution over the elements of that level
Also, maintain a distribution over the 64 most significant levels.
To facilitate updating, but unused during sampling, also maintain,
A linked list set (supports push!, delete!, in, findnext, and findprev) of levels
A pointer to the least significant tracked level (-1075 if there are fewer than 64 levels)
A vector that maps elements (integers) to their level and index in the level

To sample,
draw from the distribution over the top 64 levels and then
draw from the distribution over the elements of that level.

To add a new element at a given weight,
determine the level of that weight,
create a new level if needed,
add the element to the distribution of that level,
and update the distribution over the top 64 levels if needed.
Log the location of the new element.

To create a new level,
Push the level into the linked list set of levels.
If the level is below the least significant tracked level, that's all.
Otherwise, update the least significant tracked level and evict an element
from the distribution over the top 64 levels if necessary

To remove an element,
Lookup the location of the element
Remove the element from the distribution of its level
If the level is now empty, remove the level from the linked list set of levels
If the level is below the least significant tracked level, that's all.
Otherwise, update the least significant tracked level and add an element to the
distribution over the top 64 levels if possible
=#

struct LevelMap
    presence::BitVector
    indices::Vector{Tuple{Int, Int}}
    function LevelMap()
        presence = BitVector()
        resize!(presence, 2098)
        fill!(presence, false)
        indices = Vector{Tuple{Int, Int}}(undef, 2098)
        return new(presence, indices)
    end
end
struct EntryInfo
    presence::BitVector
    indices_out::Vector{Int16}
    indices_in::Vector{Int}
    EntryInfo() = new(BitVector(), Int16[], Int[])
end
mutable struct TrackInfo
    lastsampled_idx::Int
    lastsampled_idx_out::Int
    lastsampled_idx_in::Int
    least_significant_sampled_level::Int # The level number of the least significant tracked level
    nvalues::Int
    firstchanged::Int
    lastfull::Int
    reset_order::Int
    reset_distribution::Bool
end
struct NestedSampler
    # Used in sampling
    distribution_over_levels::SelectionSampler # A distribution over 1:64
    sampled_levels::MVector{64, Int16} # The top up to 64 levels indices
    all_levels::Vector{Tuple{UInt128, RejectionSampler}} # All the levels, in insertion order, along with their total weights

    # Not used in sampling
    sampled_level_weights::MVector{64, Float64} # The weights of the top up to 64 levels
    sampled_level_numbers::MVector{64, Int16} # The level numbers of the top up to 64 levels
    level_set::LinkedListSet # A set of which levels are non-empty (named by level number)
    level_set_map::LevelMap # A mapping from level number to index in all_levels and index in sampled_levels (or 0 if not in sampled_levels)
    entry_info::EntryInfo # A mapping from element to level number and index in that level (index in level is 0 if entry is not present)
    track_info::TrackInfo
end

NestedSampler() = NestedSampler(
    SelectionSampler(zero(MVector{64, Float64}), MVector{64, Int16}(1:64)),
    zero(MVector{64, Int16}),
    Tuple{UInt128, RejectionSampler}[],
    zero(MVector{64, Float64}),
    zero(MVector{64, Int16}),
    LinkedListSet(),
    LevelMap(),
    EntryInfo(),
    TrackInfo(0, 0, 0, -1075, 0, 1, 0, 0, true),
)

Base.rand(ns::NestedSampler, n::Integer) = rand(Random.default_rng(), ns, n)
function Base.rand(rng::AbstractRNG, ns::NestedSampler, n::Integer)
    n < 100 && return [rand(rng, ns) for _ in 1:n]
    lastfull = ns.track_info.lastfull
    totws = 0.0
    maxw = 0.0
    nvalues_sampled = 0
    for i in 1:lastfull
        w = ns.sampled_level_weights[i]
        totws += w
        maxw = ifelse(w > maxw, w, maxw)
        nvalues_sampled += length(ns.all_levels[Int(ns.sampled_levels[i])][2])
    end
    maxw/totws > 0.98 && return [rand(rng, ns) for _ in 1:n]
    nvalues_unsampled = ns.track_info.nvalues - nvalues_sampled
    r = (1-nvalues_unsampled/typemax(UInt64))^n
    n_nots = 0
    ws = @view(ns.sampled_level_weights[1:lastfull])
    inds = Vector{Int}(undef, n)
    if rand(rng) > r
        n_nots = extract_rand_multi!(rng, FallBackSampler(), ns, inds, totws, n, r)
    end
    n_each = rand(rng, Multinomial(n - n_nots, ws ./ totws))
    q = 1
    @inbounds for (level, k) in enumerate(n_each)
        bucket = ns.all_levels[Int(ns.sampled_levels[level])][2]
        f = length(bucket) <= 2048 ? randreuse : randnoreuse
        for _ in 1:k
            ti = @inline rand(rng, bucket, f)
            inds[q] = ti[2]
            q += 1
        end
    end
    shuffle!(rng, inds)
    return inds
end
Base.rand(ns::NestedSampler) = rand(Random.default_rng(), ns)
@inline function Base.rand(rng::AbstractRNG, ns::NestedSampler)
    track_info = ns.track_info
    track_info.reset_order += 1
    lastfull = track_info.lastfull
    reorder = lastfull > 8 && track_info.reset_order > 300*lastfull
    if track_info.reset_distribution || reorder 
        @inline set_cum_weights!(ns.distribution_over_levels, ns, reorder)
        track_info.reset_distribution = false
    end
    u = rand(rng, UInt64)
    if u < MAX_CUT
        return _rand(rng, ns, u, lastfull, track_info)
    else
        level_index = rand(rng, FallBackSampler(), ns, lastfull)
        level_index === 0 && return _rand(rng, ns, u, lastfull, track_info)
        j, i = rand(rng, ns.all_levels[level_index][2], randnoreuse)
        return i
    end
end
@inline function _rand(rng, ns, u, lastfull, track_info)
    v = Float64(u >>> 11) * RANDF
    level = @inline rand(rng, ns.distribution_over_levels, v, lastfull)
    j, i = @inline rand(rng, ns.all_levels[Int(ns.sampled_levels[level])][2], randnoreuse)
    track_info.lastsampled_idx = i
    track_info.lastsampled_idx_out = level
    track_info.lastsampled_idx_in = j
    return i
end

function Base.append!(ns::NestedSampler, inds::Union{AbstractRange{Int}, Vector{Int}}, 
        xs::Union{AbstractRange{Float64}, Vector{Float64}})
    ns.track_info.reset_distribution = true
    ns.track_info.reset_order += length(inds)
    ns.track_info.nvalues += length(inds)
    maxi = maximum(inds)
    l_info = lastindex(ns.entry_info.presence)
    if maxi > l_info
        newl = max(2*l_info, maxi)
        resize!(ns.entry_info.indices_out, newl)
        resize!(ns.entry_info.indices_in, newl)
        resize!(ns.entry_info.presence, newl)
        fill!(@view(ns.entry_info.presence[l_info+1:newl]), false)
    end
    for (i, x) in zip(inds, xs)
        if ns.entry_info.presence[i]
            throw(ArgumentError("Element $i is already present"))
        end
        _push!(ns, i, x)
    end
    return ns
end

@inline function Base.push!(ns::NestedSampler, i::Int, x::Float64)
    ns.track_info.reset_distribution = true
    ns.track_info.reset_order += 1
    ns.track_info.nvalues += 1
    i <= 0 && throw(ArgumentError("Elements must be positive"))
    x <= 0.0 && throw(ArgumentError("Weights must be positive"))
    l_info = lastindex(ns.entry_info.presence)
    if i > l_info
        newl = max(2*l_info, i)
        resize!(ns.entry_info.indices_out, newl)
        resize!(ns.entry_info.indices_in, newl)
        resize!(ns.entry_info.presence, newl)
        fill!(@view(ns.entry_info.presence[l_info+1:newl]), false)
    elseif ns.entry_info.presence[i]
        throw(ArgumentError("Element $i is already present"))
    end
    return _push!(ns, i, x)
end

@inline function _push!(ns::NestedSampler, i::Int, x::Float64)
    level = exponent(x)
    level_b16 = Int16(level)
    bucketw = significand(x)/2
    ns.entry_info.presence[i] = true
    if level ∉ ns.level_set
        # Log the entry
        ns.entry_info.indices_out[i] = level_b16
        ns.entry_info.indices_in[i] = 1

        # Create a new level (or revive an empty level)
        push!(ns.level_set, level)
        existing_level_indices = ns.level_set_map.presence[level+1075]
        all_levels_index = if !existing_level_indices
            level_sampler = RejectionSampler(level, i, bucketw)
            push!(ns.all_levels, (sig(x), level_sampler))
            length(ns.all_levels)
        else
            level_indices = ns.level_set_map.indices[level+1075]
            w, level_sampler = ns.all_levels[level_indices[1]]
            @assert w == 0
            @assert isempty(level_sampler)
            push!(level_sampler, i, bucketw)
            ns.all_levels[level_indices[1]] = (sig(x), level_sampler)
            level_indices[1]
        end
        ns.level_set_map.presence[level+1075] = true

        # Update the sampled levels if needed
        if level > ns.track_info.least_significant_sampled_level # we just created a sampled level
            if ns.track_info.lastfull < 64 # Add the new level to the top 64
                ns.track_info.lastfull += 1
                sl_length = ns.track_info.lastfull
                ns.sampled_levels[sl_length] = Int16(all_levels_index)
                ns.sampled_level_weights[sl_length] = x
                ns.sampled_level_numbers[sl_length] = level_b16
                ns.level_set_map.indices[level+1075] = (all_levels_index, sl_length)
                if sl_length == 64
                    ns.track_info.least_significant_sampled_level = findnext(ns.level_set, ns.track_info.least_significant_sampled_level+1)
                end
            else # Replace the least significant sampled level with the new level
                j, k = ns.level_set_map.indices[ns.track_info.least_significant_sampled_level+1075]
                ns.level_set_map.indices[ns.track_info.least_significant_sampled_level+1075] = (j, 0)
                ns.sampled_levels[k] = Int16(all_levels_index)
                ns.sampled_level_weights[k] = x
                ns.sampled_level_numbers[k] = level_b16
                ns.level_set_map.indices[level+1075] = (all_levels_index, k)
                ns.track_info.least_significant_sampled_level = findnext(ns.level_set, ns.track_info.least_significant_sampled_level+1)
                firstc = ns.track_info.firstchanged
                ns.track_info.firstchanged = ifelse(k < firstc, k, firstc)
            end
        else # created an unsampled level
            ns.level_set_map.indices[level+1075] = (all_levels_index, 0)
        end
    else # Add to an existing level
        j, k = ns.level_set_map.indices[level+1075]
        w, level_sampler = ns.all_levels[j]
        push!(level_sampler, i, bucketw)
        ns.entry_info.indices_out[i] = level_b16
        ns.entry_info.indices_in[i] = length(level_sampler)
        wn = w+sig(x)
        ns.all_levels[j] = (wn, level_sampler)

        if k != 0 # level is sampled
            ns.sampled_level_weights[k] = flot(wn, level)
            firstc = ns.track_info.firstchanged
            ns.track_info.firstchanged = ifelse(k < firstc, k, firstc)
        end
    end
    return ns
end

@inline function Base.delete!(ns::NestedSampler, i::Int)
    ns_track_info = ns.track_info
    ns_track_info.reset_distribution = true
    ns_track_info.reset_order += 1
    ns_track_info.nvalues -= 1
    if i <= 0 || i > lastindex(ns.entry_info.presence)
        throw(ArgumentError("Element $i is not present"))
    end
    if ns_track_info.lastsampled_idx == i
        level = Int(ns.sampled_level_numbers[ns_track_info.lastsampled_idx_out])
        j = ns_track_info.lastsampled_idx_in
    else
        level = Int(ns.entry_info.indices_out[i])
        j = ns.entry_info.indices_in[i]
    end
    ns_track_info.lastsampled_idx = 0
    !ns.entry_info.presence[i] && throw(ArgumentError("Element $i is not present"))
    ns.entry_info.presence[i] = false

    l, k = ns.level_set_map.indices[level+1075]
    w, level_sampler = ns.all_levels[l]
    _i, significand = level_sampler.data[j]
    @assert _i == i
    len = level_sampler.track_info.length
    moved_entry, _ = level_sampler.data[j] = level_sampler.data[len]
    level_sampler.data[len] = (0, 0.0)
    level_sampler.track_info.length -= 1
    if (len & (len-1)) == 0
        level_sampler.track_info.mask = UInt(1) << (8*sizeof(len-1) - leading_zeros(len-1)) - 1
    end
    if moved_entry != i
        @assert ns.entry_info.indices_in[moved_entry] == length(level_sampler)+1
        ns.entry_info.indices_in[moved_entry] = j
    end
    wn = w-sig(significand*exp2(level+1))
    ns.all_levels[l] = (wn, level_sampler)

    if isempty(level_sampler) # Remove a level
        delete!(ns.level_set, level)
        ns.all_levels[l] = (zero(UInt128), level_sampler) # Fixup for rounding error
        if k != 0 # Remove a sampled level
            firstc = ns.track_info.firstchanged
            ns.track_info.firstchanged = ifelse(k < firstc, k, firstc)
            replacement = findprev(ns.level_set, ns_track_info.least_significant_sampled_level-1)
            ns.level_set_map.indices[level+1075] = (l, 0)
            if isnothing(replacement) # We'll now have fewer than 64 sampled levels
                ns_track_info.least_significant_sampled_level = -1075
                sl_length = ns_track_info.lastfull
                ns_track_info.lastfull -= 1
                moved_level = ns.sampled_level_numbers[sl_length]
                if moved_level == Int16(level)
                    ns.sampled_level_weights[sl_length] = 0.0
                else
                    ns.sampled_level_numbers[k], ns.sampled_level_numbers[sl_length] = ns.sampled_level_numbers[sl_length], ns.sampled_level_numbers[k]
                    ns.sampled_levels[k], ns.sampled_levels[sl_length] = ns.sampled_levels[sl_length], ns.sampled_levels[k]
                    ns.sampled_level_weights[k] = ns.sampled_level_weights[sl_length]
                    ns.sampled_level_weights[sl_length] = 0.0
                    all_index, _l = ns.level_set_map.indices[ns.sampled_level_numbers[k]+1075]
                    @assert _l == ns.track_info.lastfull+1
                    ns.level_set_map.indices[ns.sampled_level_numbers[k]+1075] = (all_index, k)
                    all_index = ns.level_set_map.indices[ns.sampled_level_numbers[sl_length]+1075][1]
                    ns.level_set_map.indices[ns.sampled_level_numbers[sl_length]+1075] = (all_index, sl_length)
                end
            else # Replace the removed level with the replacement
                ns_track_info.least_significant_sampled_level = replacement
                all_index, _zero = ns.level_set_map.indices[replacement+1075]
                @assert _zero == 0
                ns.level_set_map.indices[replacement+1075] = (all_index, k)
                w, replacement_level = ns.all_levels[all_index]
                ns.sampled_levels[k] = Int16(all_index)
                ns.sampled_level_weights[k] = flot(w, replacement)
                ns.sampled_level_numbers[k] = replacement
            end
        end
    elseif k != 0
        ns.sampled_level_weights[k] = flot(wn, level)
        firstc = ns.track_info.firstchanged
        ns.track_info.firstchanged = ifelse(k < firstc, k, firstc)
    end
    return ns
end

struct FallBackSampler end
function Base.rand(rng::AbstractRNG, fs::FallBackSampler, ns::NestedSampler, lastfull::Int)
    totwnots = 0.0
    for i in eachindex(ns.level_set_map.indices)
        !ns.level_set_map.presence[i] && continue
        level_index, k = ns.level_set_map.indices[i]
        k !== 0 && continue
        wlevel, level_sampler = ns.all_levels[level_index]
        isempty(level_sampler) && continue
        totwnots += flot(wlevel, level_sampler.level)
    end
    totw = totwnots + ns.distribution_over_levels.p[lastfull]
    r = (typemax(UInt)/UPPER_LIMIT) * (totwnots/totw)
    rand(rng) > r && return 0
    u = rand(rng)*totwnots
    last = 0
    w = 0.0
    for i in eachindex(ns.level_set_map.indices)
        !ns.level_set_map.presence[i] && continue
        level_index, k = ns.level_set_map.indices[i]
        k !== 0 && continue
        wlevel, level_sampler = ns.all_levels[level_index]
        isempty(level_sampler) && continue
        w += flot(wlevel, level_sampler.level)
        w > u && return level_index
        last = level_index
    end
    return last
end
function extract_rand_multi!(rng::AbstractRNG, fs::FallBackSampler, ns::NestedSampler, inds, totws, n, r)
    totwnots = sum(flot(sl[1], sl[2].level) for sl in ns.all_levels 
                   if !isempty(sl[2]) && ns.level_set_map.indices[sl[2].level+1075][2] == 0)
    pnots = totwnots/(totwnots + totws)  
    n_nots = rand(rng) > (1 - (pnots^n)) / (1-r) ? rand(rng, Truncated(Binomial(n, pnots), 1, n)) : 0
    if n_nots != 0
        wnots = [flot(sl[1], sl[2].level) for sl in ns.all_levels
                 if !isempty(sl[2]) && ns.level_set_map.indices[sl[2].level+1075][2] == 0]
        n_each_nots = rand(rng, Multinomial(n_nots, wnots ./ totwnots))
        i, q = 1, n
        for sl in ns.all_levels
            isempty(sl[2]) || ns.level_set_map.indices[sl[2].level+1075][2] != 0 && continue
            bucket = sl[2]
            f = length(bucket) <= 2048 ? randreuse : randnoreuse
            for _ in 1:n_each_nots[i]
                ti = @inline rand(rng, bucket, f)
                inds[q] = ti[2]
                q -= 1
            end
            i += 1
        end
    end
    return n_nots
end

Base.in(i::Int, ns::NestedSampler) = 0 < i <= length(ns.entry_info.presence) && ns.entry_info.presence[i]

Base.isempty(ns::NestedSampler) = ns.track_info.nvalues == 0

struct SamplerIndices{I}
    ns::NestedSampler
    iter::I
end
function SamplerIndices(ns::NestedSampler)
    iter = Iterators.Flatten((Iterators.map(x -> x[1], @view(b[2].data[1:b[2].track_info.length])) for b in ns.all_levels))
    SamplerIndices(ns, iter)
end
Base.iterate(inds::SamplerIndices) = Base.iterate(inds.iter)
Base.iterate(inds::SamplerIndices, state) = Base.iterate(inds.iter, state)
Base.eltype(::Type{<:SamplerIndices}) = Int
Base.IteratorSize(::Type{<:SamplerIndices}) = Base.HasLength()
Base.length(inds::SamplerIndices) = inds.ns.track_info.nvalues

const DynamicDiscreteSampler = NestedSampler

end
