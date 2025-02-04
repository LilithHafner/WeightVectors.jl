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
