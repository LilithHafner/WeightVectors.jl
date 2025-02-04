merge_uint64(x::UInt64, y::UInt64) = UInt128(x) | (UInt128(y) << 64)
get_UInt128(m::Memory, i::Integer) = merge_uint64(m[i], m[i+1])
function verify_weights(m::Memory)
    m3 = m[3]
    for i in 5:2050
        shift = signed(2051 + m3 - i)
        weight = m[i]
        shifted_significand_sum_index = 2041 + 2i
        shifted_significand_sum = get_UInt128(m, shifted_significand_sum_index)
        expected_weight = UInt64(shifted_significand_sum<<shift)
        expected_weight += (trailing_zeros(shifted_significand_sum)+shift < 0) & (shifted_significand_sum != 0)
        @assert weight == expected_weight
    end
end

function verify_m2(m::Memory)
    @assert m[2] == findfirst(i -> i == 2051 || 5 <= i && m[i] != 0, 1:2051)
end
function verify_m4(m::Memory)
    m4 = zero(UInt64)
    for i in 5:2050
        m4 = Base.checked_add(m4, m[i])
    end
    @assert m[4] == m4
    @assert m4 == 0 || UInt64(2)^32 <= m4
end

function verify(m::Memory)
    verify_weights(m)
    verify_m2(m)
    verify_m4(m)
end
