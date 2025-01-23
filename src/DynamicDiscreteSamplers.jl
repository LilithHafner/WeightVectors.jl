module DynamicDiscreteSamplers

export DynamicDiscreteSampler, SamplerIndices

using Distributions, Random, StaticArrays

struct SelectionSampler{N}
    p::MVector{N, Float64}
    o::MVector{N, Int16}
end
function Base.rand(rng::AbstractRNG, ss::SelectionSampler, lastfull::Int)
    u = rand(rng)*ss.p[lastfull]
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
    track_info::RejectionInfo
    RejectionSampler(i, v) = new([(i, v)], RejectionInfo(1, v, zero(UInt)))
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
# Maintain a distribution over the top N levels and ignore any lower
(or maybe the top log(n) levels and treat the rest as a single level).
For each level, maintain a distribution over the elements of that level
Also, maintain a distribution over the N most significant levels.
To facilitate updating, but unused during sampling, also maintain,
A linked list set (supports push!, delete!, in, findnext, and findprev) of levels
A pointer to the least significant tracked level (-1075 if there are fewer than N levels)
A vector that maps elements (integers) to their level and index in the level

To sample,
draw from the distribution over the top N levels and then
draw from the distribution over the elements of that level.

To add a new element at a given weight,
determine the level of that weight,
create a new level if needed,
add the element to the distribution of that level,
and update the distribution over the top N levels if needed.
Log the location of the new element.

To create a new level,
Push the level into the linked list set of levels.
If the level is below the least significant tracked level, that's all.
Otherwise, update the least significant tracked level and evict an element
from the distribution over the top N levels if necessary

To remove an element,
Lookup the location of the element
Remove the element from the distribution of its level
If the level is now empty, remove the level from the linked list set of levels
If the level is below the least significant tracked level, that's all.
Otherwise, update the least significant tracked level and add an element to the
distribution over the top N levels if possible
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
    indices::Vector{Int}
    EntryInfo() = new(BitVector(), Int[])
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

@inline sig(x::Float64) = (reinterpret(UInt64, x) & Base.significand_mask(Float64)) + Base.significand_mask(Float64) + 1

@inline function flot(sg::UInt128, level::Integer)
    shift = Int64(8 * sizeof(sg) - 53 - leading_zeros(sg))
    x = (sg >>= shift) % UInt64
    exp = level + shift + 1022
    reinterpret(Float64, x + (exp << 52))
end

struct NestedSampler{N}
    # Used in sampling
    distribution_over_levels::SelectionSampler{N} # A distribution over 1:N
    sampled_levels::MVector{N, Int16} # The top up to N levels indices
    all_levels::Vector{Tuple{UInt128, RejectionSampler}} # All the levels, in insertion order, along with their total weights

    # Not used in sampling
    sampled_level_weights::MVector{N, Float64} # The weights of the top up to N levels
    sampled_level_numbers::MVector{N, Int16} # The level numbers of the top up to N levels
    level_set::LinkedListSet # A set of which levels are non-empty (named by level number)
    level_set_map::LevelMap # A mapping from level number to index in all_levels and index in sampled_levels (or 0 if not in sampled_levels)
    entry_info::EntryInfo # A mapping from element to level number and index in that level (index in level is 0 if entry is not present)
    track_info::TrackInfo
end

NestedSampler() = NestedSampler{64}()
NestedSampler{N}() where N = NestedSampler{N}(
    SelectionSampler(zero(MVector{N, Float64}), MVector{N, Int16}(1:N)),
    zero(MVector{N, Int16}),
    Tuple{UInt128, RejectionSampler}[],
    zero(MVector{N, Float64}),
    zero(MVector{N, Int16}),
    LinkedListSet(),
    LevelMap(),
    EntryInfo(),
    TrackInfo(0, 0, 0, -1075, 0, 1, 0, 0, true),
)

Base.rand(ns::NestedSampler, n::Integer) = rand(Random.default_rng(), ns, n)
function Base.rand(rng::AbstractRNG, ns::NestedSampler, n::Integer)
    n < 100 && return [rand(rng, ns) for _ in 1:n]
    lastfull = ns.track_info.lastfull
    ws = @view(ns.sampled_level_weights[1:lastfull])
    totw = sum(ws)
    maxw = maximum(ws)
    maxw/totw > 0.98 && return [rand(rng, ns) for _ in 1:n]
    n_each = rand(rng, Multinomial(n, ws ./ totw))
    inds = Vector{Int}(undef, n)
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
    level = @inline rand(rng, ns.distribution_over_levels, lastfull)
    j, i = @inline rand(rng, ns.all_levels[Int(ns.sampled_levels[level])][2], randnoreuse)
    track_info.lastsampled_idx = i
    track_info.lastsampled_idx_out = level
    track_info.lastsampled_idx_in = j
    return i
end

function Base.append!(ns::NestedSampler{N}, inds::Union{AbstractRange{Int}, Vector{Int}},
        xs::Union{AbstractRange{Float64}, Vector{Float64}}) where N
    ns.track_info.reset_distribution = true
    ns.track_info.reset_order += length(inds)
    ns.track_info.nvalues += length(inds)
    maxi = maximum(inds)
    l_info = lastindex(ns.entry_info.presence)
    if maxi > l_info
        newl = max(2*l_info, maxi)
        resize!(ns.entry_info.indices, newl)
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

@inline function Base.push!(ns::NestedSampler{N}, i::Int, x::Float64) where N
    ns.track_info.reset_distribution = true
    ns.track_info.reset_order += 1
    ns.track_info.nvalues += 1
    i <= 0 && throw(ArgumentError("Elements must be positive"))
    l_info = lastindex(ns.entry_info.presence)
    if i > l_info
        newl = max(2*l_info, i)
        resize!(ns.entry_info.indices, newl)
        resize!(ns.entry_info.presence, newl)
        fill!(@view(ns.entry_info.presence[l_info+1:newl]), false)
    elseif ns.entry_info.presence[i]
        throw(ArgumentError("Element $i is already present"))
    end
    return _push!(ns, i, x)
end

@inline function _push!(ns::NestedSampler{N}, i::Int, x::Float64) where N
    level = exponent(x)
    level_b16 = Int16(level)
    bucketw = significand(x)/2
    ns.entry_info.presence[i] = true
    if level ∉ ns.level_set
        # Log the entry
        ns.entry_info.indices[i] = 4096 + level + 1075

        # Create a new level (or revive an empty level)
        push!(ns.level_set, level)
        existing_level_indices = ns.level_set_map.presence[level+1075]
        all_levels_index = if !existing_level_indices
            level_sampler = RejectionSampler(i, bucketw)
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
            if ns.track_info.lastfull < N # Add the new level to the top 64
                ns.track_info.lastfull += 1
                sl_length = ns.track_info.lastfull
                ns.sampled_levels[sl_length] = Int16(all_levels_index)
                ns.sampled_level_weights[sl_length] = x
                ns.sampled_level_numbers[sl_length] = level_b16
                ns.level_set_map.indices[level+1075] = (all_levels_index, sl_length)
                if sl_length == N
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
        ns.entry_info.indices[i] = length(level_sampler) << 12 + level + 1075
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
        c = ns.entry_info.indices[i]
        level = c & 4095 - 1075
        j = (c - level - 1075) >> 12
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
        @assert ns.entry_info.indices[moved_entry] == (length(level_sampler)+1) << 12 + level + 1075
        ns.entry_info.indices[moved_entry] = j << 12 + level + 1075
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
            if isnothing(replacement) # We'll now have fewer than N sampled levels
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

# Take Two!
# - Exact
# - O(1) in theory
# - Fast in practice

#=

levels are powers of two. Each level has a true weight which is the sum of the (Float64) weights of
the elements in that level and is represented as a UInt128 which is the sum of the significands of that level (exponent stored implicitly).
Each level also has a an approximate weight which is represented as a UInt64 with an implicit "<< level0" where
level0 is a constant maintained by the sampler so that the sum of the approximate weights is less
than 2^64 and greator than 2^32. The sum of the approximate weights and index of the highest level
are also maintained.

To select a level, pick a random number in Base.OneTo(sum of approximate weights) and find that
level with linear search

maintain the total weight

=#

abstract type Weights end
struct FixedSizeWeights <: Weights
    m::Memory{UInt64}
    global _FixedSizeWeights
    _FixedSizeWeights(m::Memory{UInt64}) = new(m)
end
struct SemiResizableWeights <: Weights
    m::Memory{UInt64}
    SemiResizableWeights(w::FixedSizeWeights) = new(w.m)
end
mutable struct ResizableWeights <: Weights
    m::Memory{UInt64}
    ResizableWeights(w::FixedSizeWeights) = new(w.m)
end

## Standard memory layout: (TODO: add alternative layout for small cases)

# <memory_length::Int>
# length::Int
# max_level::Int # absolute pointer to the first element of level weights that is nonzero
# shift::Int level weights are euqal to shifted_significand_sums<<(exponent_bits+shift) rounded up
# sum(level weights)::UInt64
# level weights::[UInt64 2046] # earlier is higher. first is exponent bits 0x7fe, last is exponent bits 0x001. Subnormal are not supported.
# shifted_significand_sums::[UInt128 2046] # sum of significands shifted by 11 bits to the left with their leading 1s appended (the maximum significand contributes typemax(UInt64))
# level location info::[NamedTuple{posm2::Int, length::Int} 2046] indexes into sub_weights, posm2 is absolute into m.

# gc info:
# next_free_space::Int (used to re-allocate) <index 10235>
# 32 unused bits
# level allocated length::[UInt8 2046] (2^x is implied)

# edit_map (maps index to current location in sub_weights)::[pos::Int, exponent::Int] (zero means zero; fixed location, always at the start. Force full realloc when it OOMs. exponent could be UInt11, lots of wasted bits)

# sub_weights (woven with targets)::[[(2^63-significand<<11)::UInt64, target::Int}]] aka 2^64-significand_with_leading_1<<11

## Initial API:

# setindex!, getindex, resize! (auto-zeros), scalar rand
# Trivial extensions:
# push!, delete!

Base.rand(rng::AbstractRNG, w::Weights) = _rand(rng, w.m)
Base.getindex(w::Weights, i::Int) = _getindex(w.m, i)
Base.setindex!(w::Weights, v, i::Int) = (_setindex!(w.m, Float64(v), i); w)

#=@inbounds=# function _rand(rng::AbstractRNG, m::Memory{UInt64})

    @label reject

    # Select level
    x = rand(rng, Base.OneTo(m[4]))
    i = m[2]
    while #=i < 2046+4=# true
        mi = m[i]
        x <= mi && break
        x -= mi
    end

    # Low-probability rejection to improve accuracy from very close to perfect
    if x == mi # mi is the weight rounded up. If they are equal than we should refine futher and possibly reject. This branch is very uncommon and still O(1); constant factors don't matter here.
        # significand_sum::UInt128 = ...
        # weight::UInt64 = mi = ceil(significand_sum<<*(exponent_bits+shift))...
        # rejection_p = ceil(significand_sum<<*(exponent_bits+shift)) - true(ceil(significand_sum<<*(exponent_bits+shift)))
        # rejection_p = 1-rem(significand_sum<<*(exponent_bits+shift))
        # acceptance_p = rem(significand_sum<<*(exponent_bits+shift))
        # acceptance_p = significand_sum<<*(exponent_bits+shift) & ...00000.111111...
        j = 2i+2041
        exponent_bits = 0x7fe+5-i
        shift = exponent_bits + m[3]
        significand_sum = reinterpret(UInt128, (m[j], m[j+1]))
        while true
            x = rand(rng, UInt64)
            # p_stage = significand_sum << shift & ...00000.111111...64...11110000
            target = significand_sum << (shift + 64) % UInt64
            x > target && @goto reject
            x < target && break
            shift += 64
            shift >= 0 && break
        end
    end

    # Lookup level info
    j = 2i + 6133
    posm2 = m[j]
    len = m[j+1]

    # Sample within level
    while true
        k = 2rand(rng, Base.OneTo(len))+posm2
        rand(rng, UInt64) > m[k] && return Int(signed(m[k+1]))
    end
end

function _getindex(m::Memory{UInt64}, i::Int)
    @boundscheck 1 <= i <= m[1] || throw(BoundsError(FixedSizeWeights(m), i))
    j = 2i + 10490
    pos = m[j]
    pos == 0 && return 0.0
    exponent = m[j+1]
    weight = m[pos+1]
    reinterpret(Float64, exponent | (weight >> 12))
end

function _setindex!(m::Memory, v::Float64, i::Int)
    @boundscheck 1 <= i <= m[1] || throw(BoundsError(FixedSizeWeights(m), i))
    uv = reinterpret(UInt64, v)
    if uv == 0
        _set_to_zero!(m, i)
        return
    end
    0x0010000000000000 <= uv <= 0x7fefffffffffffff || throw(DomainError(v, "Invalid weight")) # Excludes subnormals

    # Find the entry's pos in the edit map table
    j = 2i + 10490
    pos = m[j]
    if pos == 0
        _set_from_zero!(m, v, i::Int)
    else
        _set_nonzero!(m, v, i::Int)
    end
end

function _set_nonzero!(m, v, i)
    # TODO for performance: join these two opperations
    _set_to_zero!(m, i)
    _set_nonzero!(m, v, i)
end

function _set_from_zero!(m::Memory, v::Float64, i::Int)
    uv = reinterpret(UInt, v)
    j = 2i + 10490
    @assert m[j] == 0

    exponent = uv & Base.exponent_mask(Float64)
    m[j+1] = exponent

    # update group total weight and total weight
    shifted_significand_sum_index = get_shifted_significand_sum_index(exponent)
    shifted_significand_sum = get_UInt128(m, shifted_significand_sum_index)
    shifted_significand = (uv & Base.significand_mask(Float64)) << 11
    shifted_significand_sum += -shifted_significand # the negation overflows and the += does not so these do not join to -=.
    set_UInt128!(m, shifted_significand_sum, shifted_significand_sum_index)
    update_weights!(m, exponent, shifted_significand_sum)

    # lookup the group by exponent and bump length
    group_length_index = shifted_significand_sum_index + 2*2046 + 1
    group_posm2 = m[group_length_index-1]
    group_length = m[group_length_index]+1
    m[group_length_index] = group_length
    allocs_index,allocs_subindex = get_alloced_indices(exponent)
    allocs_chunk = m[allocs_index]
    log2_allocated_size = allocs_chunk >> allocs_subindex % UInt8
    allocated_size = 1<<log2_allocated_size

    # if there is not room in the group, shift and expand
    if group_length > allocated_size
        next_free_space = m[10235]
        # if at end already, simply extend the allocation # TODO see if removing this optimization is problematic; TODO verify the optimization is triggering
        if next_free_space == group_posm2+2group_length
            # expand the allocated size and bump next_free_space
            log2_new_allocated_size = log2_allocated_size+1
            new_chunk = allocs_chunk + 1 << allocs_subindex
            m[allocs_index] = new_chunk
            m[10235] = next_free_space+allocated_size
        else # move and reallocate
            new_next_free_space = next_free_space+allocated_size<<1
            if new_next_free_space > length(m)+1 # out of space; compact. TODO for perf, consider resizing at this time slightly eagerly?
                firstindex_of_compactee = 2m[1] + 10493
                next_free_space = compact!(m, firstindex_of_compactee, m, firstindex_of_compactee)
                new_next_free_space = next_free_space+allocated_size<<1
                @assert new_next_free_space < length(m)+1 # After compaction there should be room TODO for perf, delete this
            end
            # TODO for perf, try removing the moveie before compaction (tricky: where to store that info?)
            # TODO make this whole alg dry, but only after setting up robust benchmarks in CI

            # expand the allocated size and bump next_free_space
            log2_new_allocated_size = log2_allocated_size+1
            new_chunk = allocs_chunk + 1 << allocs_subindex
            m[allocs_index] = new_chunk
            m[10235] = new_next_free_space

            # Copy the group to new location
            unsafe_copyto!(m, group_posm2+2, m, next_free_space, group_length-1)

            # Adjust the pos entries in edit_map (bad memory order TODO: consider unzipping edit map to improve locality here)
            delta = next_free_space-group_posm2+2
            for k in 0:group_length-2
                target = src[next_free_space+2k]
                j = 2target + 10485
                dst[j] += delta
            end

            group_posm2 = next_free_space-2
        end
    end

    # insert the element into the group
    group_lastpos = group_posm2+2group_length
    m[group_lastpos] = shifted_significand
    m[group_lastpos+1] = i

    # log the insertion location in the edit map
    m[j] = group_lastpos+1

    nothing
end

get_shifted_significand_sum_index(exponent::UInt64) = 5 + 3*2046 + 512 - exponent >> 51
get_UInt128(m::Memory, i::Integer) = reinterpret(UInt128, (m[i], m[i+1]))
set_UInt128!(m::Memory, v::UInt128, i::Integer) = m[i:i+1] .= reinterpret(Tuple{UInt64, UInt64}, v)
"computes shifted_significand_sum<<(exponent_bits+shift) rounded up"
function compute_weight(m::Memory, exponent::UInt64, shifted_significand_sum::UInt128)
    shift = (exponent >> 52 + m[3])
    weight = UInt64(shifted_significand_sum<<shift) # TODO for perf: change to % UInt64
    # round up
    weight += (trailing_zeros(shifted_significand_sum)+shift < 0) & (shifted_significand_sum != 0)
    weight
end
function update_weights!(m::Memory, exponent::UInt64, shifted_significand_sum::UInt128)
    weight = compute_weight(m, exponent, shifted_significand_sum)
    weight_index = 5 + 0x7fe - exponent >> 52
    old_weight = m[weight_index]
    m[weight_index] = weight
    m[4] += old_weight - weight
end

get_alloced_indices(exponent::UInt64) = 10747 - exponent >> 54, exponent >> 49 & 0x18

function _set_to_zero!(m::Memory, i::Int)
    # Find the entry's pos in the edit map table
    j = 2i + 10490
    pos = m[j]
    pos == 0 && return # if the entry is already zero, return
    # set the entry to zero (no need to zero the exponent)
    m[j] = 0
    exponent = m[j+1]

    # update group total weight and total weight
    shifted_significand_sum_index = get_shifted_significand_sum_index(exponent)
    shifted_significand_sum = get_UInt128(m, shifted_significand_sum_index)
    shifted_significand = m[pos]
    shifted_significand_sum -= -shifted_significand # the negation overflows and the -= does not so these do not cancel.
    set_UInt128!(m, shifted_significand_sum, shifted_significand_sum_index)
    update_weights!(m, exponent, shifted_significand_sum)

    # lookup the group by exponent
    group_length_index = shifted_significand_sum_index + 2*2046 + 1
    group_posm2 = src[group_length_index-1]
    group_length = src[group_length_index]
    group_lastpos = group_posm2+2group_length

    # shift the last element of the group into the spot occupied by the removed element
    m[pos] = m[group_lastpos]
    m[pos+1] = m[group_lastpos+1]

    # shrink the group
    src[group_length_index] = group_length-1 # no need to zero group entries

    nothing
end


ResizableWeights(len::Integer) = ResizableWeights(FixedSizeWeights(len))
SemiResizableWeights(len::Integer) = SemiResizableWeights(FixedSizeWeights(len))
function FixedSizeWeights(len::Integer)
    m = Memory{UInt64}(undef, allocated_memory(len))
    m[3:10747+2len] .= 0 # metadata and edit map need to be zeroed but the bulk does not
    m[1] = len
    m[2] = 2050
    # m[3]...?
    m[10235] = 10748+2len
    _FixedSizeWeights(m)
end
allocated_memory(length::Integer) = 10747 + 8*length
length_from_memory(allocated_memory::Integer) = (allocated_memory-10747) >> 3 # Int((allocated_memory-10747)/8)

function Base.resize!(w::Union{SemiResizableWeights, ResizableWeights}, len::Integer)
    m = w.m
    old_len = m[1]
    if len > old_len
        am = allocated_memory(len)
        if am > length(m)
            w isa SemiResizableWeights && throw(ArgumentError("Cannot increase the size of a SemiResizableWeights above its original allocated size. Try using a ResizableWeights instead."))
            _resize!(w, len)
        else
            m[1] = len
        end
    else
        w[len+1:old_len] .= 0 # This is a necessary but highly nontrivial operation
        m[1] = len
    end
    w
end
"Reallocate w with the size len, compacting w into that new memory"
function _resize!(w::ResizableWeights, len::Integer)
    m = w.m
    old_len = m[1]
    m2 = Memory{UInt64}(undef, allocated_memory(len))
    m2[1] = len
    if len > old_len # grow
        unsafe_copyto!(m2, 2, m, 2, 2old_len + 10491)
        m2[2old_len + 10493:2len + 10492] .= 0
    else # shrink
        unsafe_copyto!(m2, 2, m, 2, 2len + 10491)
    end

    compact!(new_m, 2len + 10493, m, 2old_len + 10493)
    w.m = new_m
    w
end

function compact!(dst::Memory{UInt64}, dst_i::Int, src::Memory{UInt64}, src_i::Int)
    # len = src[1]
    next_free_space = src[10235]
    # dst_i = src_i = 2len + 10493

    while src_i < next_free_space

        # Skip over abandoned groups
        target = signed(src[src_i])
        while target < 0
            src_i -= target
            target = signed(src[src_i])
        end

        # Trace an element of the group back to the edit info table to find the group id
        j = 2target + 10485
        exponent = m[j+1]

        # Lookup the group in the group location table to find its length (performance optimization for copying, necesary to decide new allocated size)
        # exponent of 0x7fe0000000000000 is index 6+3*2046
        # exponent of 0x0010000000000000 is index 4+5*2046
        group_length_index = 6 + 5*2046 + 512 - (exponent >> 51)
        group_length = src[group_length_index]

        # Lookup the allocated size (an alternative to scanning for the next nonzero, needed because we are setting allocated size)
        # exponent of 0x7fe0000000000000 is index 6+5*2046, 2
        # exponent of 0x7fd0000000000000 is index 6+5*2046, 1
        # exponent of 0x0040000000000000 is index 5+5*2046+512, 0
        # exponent of 0x0030000000000000 is index 5+5*2046+512, 3
        # exponent of 0x0020000000000000 is index 5+5*2046+512, 2
        # exponent of 0x0010000000000000 is index 5+5*2046+512, 1
        # allocs_index = 5+5*2046+512 - (exponent >> 54), (exponent >> 52) & 0x3
        # allocated_size = 2 << ((src[allocs_index[1]] >> (8allocs_index[1])) % UInt8)
        allocs_index,allocs_subindex = get_alloced_indices(exponent)
        allocs_chunk = src[allocs_index]
        log2_allocated_size = allocs_chunk >> allocs_subindex % UInt8
        log2_new_allocated_size = Base.top_set_bit(group_length-1)
        new_chunk = allocs_chunk ⊻ (log2_new_allocated_size ⊻ log2_allocated_size) << allocs_subindex
        src[allocs_index] = new_chunk

        # Copy the group to a compacted location
        unsafe_copyto!(dst, dst_i-1, src, src_i-1, 2group_length)

        # Adjust the pos entries in edit_map (bad memory order TODO: consider unzipping edit map to improve locality here)
        delta = dst_i-src_i
        dst[j] += delta
        for k in 1:group_length-1
            target = src[src_i+2k]
            j = 2target + 10485
            dst[j] += delta
        end

        # Advance indices
        src_i += 1<<log2_allocated_size
        dst_i += 1<<log2_new_allocated_size
    end
    dst[10235] = dst_i
end

end