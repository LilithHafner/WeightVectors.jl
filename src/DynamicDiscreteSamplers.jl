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
end
struct RejectionSampler
    data::Vector{Tuple{Int, Float64}}
    track_info::RejectionInfo
    RejectionSampler(i, v) = new([(i, v)], RejectionInfo(1, v))
end
function Random.rand(rng::AbstractRNG, rs::RejectionSampler)
    len = rs.track_info.length
    mask = UInt64(1) << Base.top_set_bit(len - 1) - 1 # assumes length(data) is the power of two next after (or including) rs.length[]
    maxw = rs.track_info.maxw
    while true
        u = rand(rng, UInt)
        i = u & mask + 1
        i > len && continue
        res, x = rs.data[i]
        rand(rng) * maxw < x && return (i, res) # TODO: consider reusing random bits from u; a previous test revealed no perf improvement from doing this
    end
end
function Base.push!(rs::RejectionSampler, i, x)
    len = rs.track_info.length += 1
    len > length(rs.data) && resize!(rs.data, length(rs.data)+len-1)
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
    indices::Vector{Tuple{Int, Int}}
    function EntryInfo()
        presence = BitVector()
        indices = Tuple{Int, Int}[]
        return new(presence, indices)
    end
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
    sampled_levels::MVector{N, Int16} # The top up to 64 levels indices
    all_levels::Vector{Tuple{UInt128, RejectionSampler}} # All the levels, in insertion order, along with their total weights

    # Not used in sampling
    sampled_level_weights::MVector{N, Float64} # The weights of the top up to N levels
    sampled_level_numbers::MVector{N, Int16} # The level numbers of the top up to N levels TODO: consider merging with sampled_levels_weights
    level_set::LinkedListSet # A set of which levels are present (named by level number)
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
        for _ in 1:k
            ti = @inline rand(rng, bucket)
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
    j, i = @inline rand(rng, ns.all_levels[Int(ns.sampled_levels[level])][2])
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
    l_info = lastindex(ns.entry_info.indices)
    if maxi > l_info
        resize!(ns.entry_info.indices, maxi)
        resize!(ns.entry_info.presence, maxi)
        fill!(@view(ns.entry_info.presence[l_info+1:maxi]), false)
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
    l_info = lastindex(ns.entry_info.indices)
    if i > l_info
        resize!(ns.entry_info.indices, i)
        resize!(ns.entry_info.presence, i)
        fill!(@view(ns.entry_info.presence[l_info+1:i]), false)
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
        ns.entry_info.indices[i] = (level, 1)

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
        ns.entry_info.indices[i] = (level, length(level_sampler))
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
    if i <= 0 || i > lastindex(ns.entry_info.indices)
        throw(ArgumentError("Element $i is not present"))
    end
    if ns_track_info.lastsampled_idx == i
        level = Int(ns.sampled_level_numbers[ns_track_info.lastsampled_idx_out])
        j = ns_track_info.lastsampled_idx_in
    else
        level, j = ns.entry_info.indices[i]
    end
    ns_track_info.lastsampled_idx = 0
    !ns.entry_info.presence[i] && throw(ArgumentError("Element $i is not present"))
    ns.entry_info.presence[i] = false

    l, k = ns.level_set_map.indices[level+1075]
    w, level_sampler = ns.all_levels[l]
    _i, significand = level_sampler.data[j]
    @assert _i == i
    moved_entry, _ = level_sampler.data[j] = level_sampler.data[level_sampler.track_info.length]
    level_sampler.data[level_sampler.track_info.length] = (0, 0.0)
    level_sampler.track_info.length -= 1
    if moved_entry != i
        @assert ns.entry_info.indices[moved_entry] == (level, length(level_sampler)+1)
        ns.entry_info.indices[moved_entry] = (level, j)
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
