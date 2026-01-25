//! AVX-512 implementation of filter_vec_in_place
//! Based on: https://quickwit.io/blog/simd-range
use std::arch::x86_64::{
    __m512i, _mm512_add_epi32 as op_add, _mm512_cmpge_epi32_mask, _mm512_cmple_epi32_mask,
    _mm512_loadu_epi32 as load_unaligned, _mm512_mask_compressstoreu_epi32 as compress_store,
    _mm512_set1_epi32 as set1, _mm512_xor_epi32 as op_xor,
};
use std::ops::RangeInclusive;

const NUM_LANES: usize = 16;
const INITIAL_IDS: __m512i = from_u32x16([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
const SHIFT: __m512i = from_u32x16([NUM_LANES as u32; NUM_LANES]);

const HIGHEST_BIT: u32 = 1 << 31;

#[inline]
fn u32_to_i32(val: u32) -> i32 {
    (val ^ HIGHEST_BIT) as i32
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn u32_to_i32_avx512(vals: __m512i) -> __m512i {
    const HIGHEST_BIT_MASK: __m512i = from_u32x16([HIGHEST_BIT; NUM_LANES]);
    op_xor(vals, HIGHEST_BIT_MASK)
}

pub fn filter_vec_in_place(range: RangeInclusive<u32>, offset: u32, output: &mut Vec<u32>) {
    let range_i32: RangeInclusive<i32> = u32_to_i32(*range.start())..=u32_to_i32(*range.end());
    let num_words = output.len() / NUM_LANES;
    let mut output_len = unsafe {
        filter_vec_avx512_aux(
            output.as_ptr() as *const __m512i,
            range_i32,
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

#[target_feature(enable = "avx512f")]
unsafe fn filter_vec_avx512_aux(
    mut input: *const __m512i,
    range: RangeInclusive<i32>,
    output: *mut u32,
    offset: u32,
    num_words: usize,
) -> usize {
    let mut output_tail = output;
    let range_start = set1(*range.start());
    let range_end = set1(*range.end());
    let offset_vec = set1(offset as i32);
    let mut ids = op_add(INITIAL_IDS, offset_vec);
    for _ in 0..num_words {
        let word = load_unaligned(input as *const i32);
        let word_transformed = u32_to_i32_avx512(word);
        let ge_start = _mm512_cmpge_epi32_mask(word_transformed, range_start);
        let le_end = _mm512_cmple_epi32_mask(word_transformed, range_end);
        let keeper_mask: u16 = ge_start & le_end;
        compress_store(output_tail as *mut i32, keeper_mask, ids);
        let added_len = keeper_mask.count_ones();
        output_tail = output_tail.offset(added_len as isize);
        ids = op_add(ids, SHIFT);
        input = input.offset(1);
    }
    output_tail.offset_from(output) as usize
}

union U32x16 {
    vector: __m512i,
    vals: [u32; NUM_LANES],
}

const fn from_u32x16(vals: [u32; NUM_LANES]) -> __m512i {
    unsafe { U32x16 { vals }.vector }
}
