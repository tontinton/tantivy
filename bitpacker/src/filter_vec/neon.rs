//! ARM NEON implementation of filter_vec_in_place
//! Based on: https://quickwit.io/blog/simd-range
//!
//! NEON operates on 128-bit registers (4x u32), similar to SSE on x86.
//! Unlike AVX-512, NEON doesn't have a compress instruction, so we use
//! a lookup table approach similar to AVX2.
use std::arch::aarch64::{
    uint32x4_t, vaddq_u32, vandq_u32, vcgeq_u32, vcleq_u32, vdupq_n_u32, veorq_u32, vgetq_lane_u32,
    vld1q_u32, vqtbl1q_u8, vreinterpretq_u32_u8, vreinterpretq_u8_u32, vst1q_u32,
};
use std::ops::RangeInclusive;

const NUM_LANES: usize = 4;
const HIGHEST_BIT: u32 = 1 << 31;

#[inline]
fn u32_to_i32_scalar(val: u32) -> u32 {
    val ^ HIGHEST_BIT
}

#[inline]
unsafe fn u32_to_i32_neon(vals: uint32x4_t) -> uint32x4_t {
    let highest_bit_mask = vdupq_n_u32(HIGHEST_BIT);
    veorq_u32(vals, highest_bit_mask)
}

pub fn filter_vec_in_place(range: RangeInclusive<u32>, offset: u32, output: &mut Vec<u32>) {
    let range_transformed = u32_to_i32_scalar(*range.start())..=u32_to_i32_scalar(*range.end());
    let num_words = output.len() / NUM_LANES;
    let mut output_len = unsafe {
        filter_vec_neon_aux(
            output.as_ptr(),
            range_transformed,
            output.as_mut_ptr(),
            offset,
            num_words,
        )
    };
    let remainder_start = num_words * NUM_LANES;
    for i in remainder_start..output.len() {
        let val = output[i];
        output[output_len] = offset + i as u32;
        output_len += if range.contains(&val) { 1 } else { 0 };
    }
    output.truncate(output_len);
}

#[target_feature(enable = "neon")]
unsafe fn filter_vec_neon_aux(
    mut input: *const u32,
    range: RangeInclusive<u32>,
    output: *mut u32,
    offset: u32,
    num_words: usize,
) -> usize {
    let mut output_tail = output;
    let range_start = vdupq_n_u32(*range.start());
    let range_end = vdupq_n_u32(*range.end());
    let mut ids = from_u32x4([offset, offset + 1, offset + 2, offset + 3]);
    let shift = vdupq_n_u32(NUM_LANES as u32);

    for _ in 0..num_words {
        let word = vld1q_u32(input);
        let word_transformed = u32_to_i32_neon(word);
        let keeper_bitset = compute_filter_bitset(word_transformed, range_start, range_end);
        let added_len = keeper_bitset.count_ones();
        let filtered_ids = compact(ids, keeper_bitset);
        vst1q_u32(output_tail, filtered_ids);
        output_tail = output_tail.offset(added_len as isize);
        ids = vaddq_u32(ids, shift);
        input = input.add(NUM_LANES);
    }
    output_tail.offset_from(output) as usize
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn compute_filter_bitset(
    val: uint32x4_t,
    range_start: uint32x4_t,
    range_end: uint32x4_t,
) -> u8 {
    let ge_start = vcgeq_u32(val, range_start);
    let le_end = vcleq_u32(val, range_end);
    let inside = vandq_u32(ge_start, le_end);
    mask_to_bitset(inside)
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn mask_to_bitset(mask: uint32x4_t) -> u8 {
    let lane0 = (vgetq_lane_u32::<0>(mask) != 0) as u8;
    let lane1 = (vgetq_lane_u32::<1>(mask) != 0) as u8;
    let lane2 = (vgetq_lane_u32::<2>(mask) != 0) as u8;
    let lane3 = (vgetq_lane_u32::<3>(mask) != 0) as u8;
    lane0 | (lane1 << 1) | (lane2 << 2) | (lane3 << 3)
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn compact(data: uint32x4_t, mask: u8) -> uint32x4_t {
    let perm = MASK_TO_PERMUTATION[mask as usize & 0x0F];
    let data_bytes = vreinterpretq_u8_u32(data);
    let shuffled = vqtbl1q_u8(data_bytes, perm);
    vreinterpretq_u32_u8(shuffled)
}

#[inline]
unsafe fn from_u32x4(vals: [u32; NUM_LANES]) -> uint32x4_t {
    vld1q_u32(vals.as_ptr())
}

union U8x16 {
    vector: std::arch::aarch64::uint8x16_t,
    vals: [u8; 16],
}

const fn from_u8x16(vals: [u8; 16]) -> std::arch::aarch64::uint8x16_t {
    unsafe { U8x16 { vals }.vector }
}

const fn permutation_for_mask(mask: u8) -> [u8; 16] {
    let mut result = [0u8; 16];
    let mut write_pos = 0usize;
    let mut lane = 0u8;
    while lane < 4 {
        if (mask >> lane) & 1 == 1 {
            let byte_offset = lane * 4;
            result[write_pos * 4] = byte_offset;
            result[write_pos * 4 + 1] = byte_offset + 1;
            result[write_pos * 4 + 2] = byte_offset + 2;
            result[write_pos * 4 + 3] = byte_offset + 3;
            write_pos += 1;
        }
        lane += 1;
    }
    while write_pos < 4 {
        result[write_pos * 4] = 0;
        result[write_pos * 4 + 1] = 0;
        result[write_pos * 4 + 2] = 0;
        result[write_pos * 4 + 3] = 0;
        write_pos += 1;
    }
    result
}

const MASK_TO_PERMUTATION: [std::arch::aarch64::uint8x16_t; 16] = [
    from_u8x16(permutation_for_mask(0)),
    from_u8x16(permutation_for_mask(1)),
    from_u8x16(permutation_for_mask(2)),
    from_u8x16(permutation_for_mask(3)),
    from_u8x16(permutation_for_mask(4)),
    from_u8x16(permutation_for_mask(5)),
    from_u8x16(permutation_for_mask(6)),
    from_u8x16(permutation_for_mask(7)),
    from_u8x16(permutation_for_mask(8)),
    from_u8x16(permutation_for_mask(9)),
    from_u8x16(permutation_for_mask(10)),
    from_u8x16(permutation_for_mask(11)),
    from_u8x16(permutation_for_mask(12)),
    from_u8x16(permutation_for_mask(13)),
    from_u8x16(permutation_for_mask(14)),
    from_u8x16(permutation_for_mask(15)),
];
