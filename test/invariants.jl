isdefined(@__MODULE__, :Memory) || const Memory = Vector # Compat for Julia < 1.11
_get_UInt128(m::Memory, i::Integer) = UInt128(m[i]) | (UInt128(m[i+1]) << 64)
function verify_weights(m::Memory)
    m3 = m[3]
    for i in 5:2050
        shift = signed(i - 4 + m3)
        weight = m[i]
        shifted_significand_sum_index = 2041 + 2i
        shifted_significand_sum = _get_UInt128(m, shifted_significand_sum_index)
        expected_weight = UInt64(shifted_significand_sum<<shift)
        expected_weight += (shifted_significand_sum != 0)
        @assert weight == expected_weight
    end
end

function verify_m2(m::Memory)
    @assert m[2] == findlast(i -> i == 4 || m[i] != 0, 1:2050)
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
