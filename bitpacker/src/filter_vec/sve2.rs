//! ARM SVE2 implementation of filter_vec_in_place
//! Based on: https://quickwit.io/blog/simd-range
//!
//! SVE2 (Scalable Vector Extension 2) provides the COMPACT instruction which
//! compresses active elements to the front of a vector, similar to AVX-512's
//! VPCOMPRESSD. This eliminates the need for lookup tables used in NEON.
//!
//! SVE vectors are scalable-length (128 to 2048 bits), so we query the vector
//! length at runtime using svcntw() and process that many u32s per iteration.

use std::arch::asm;
use std::ops::RangeInclusive;

const HIGHEST_BIT: u32 = 1 << 31;

#[inline]
fn u32_to_i32_scalar(val: u32) -> u32 {
    val ^ HIGHEST_BIT
}

pub fn filter_vec_in_place(range: RangeInclusive<u32>, offset: u32, output: &mut Vec<u32>) {
    unsafe { filter_vec_in_place_impl(range, offset, output) }
}

#[target_feature(enable = "sve2")]
unsafe fn filter_vec_in_place_impl(range: RangeInclusive<u32>, offset: u32, output: &mut Vec<u32>) {
    let range_transformed = u32_to_i32_scalar(*range.start())..=u32_to_i32_scalar(*range.end());

    let vl = sve_cntw();
    if vl == 0 {
        return;
    }

    let num_words = output.len() / vl;
    let mut output_len = filter_vec_sve_aux(
        output.as_ptr(),
        range_transformed,
        output.as_mut_ptr(),
        offset,
        num_words,
        vl,
    );

    let remainder_start = num_words * vl;
    for i in remainder_start..output.len() {
        let val = output[i];
        output[output_len] = offset + i as u32;
        output_len += if range.contains(&val) { 1 } else { 0 };
    }
    output.truncate(output_len);
}

#[inline]
#[target_feature(enable = "sve2")]
unsafe fn sve_cntw() -> usize {
    let count: usize;
    asm!(
        "cntw {out}",
        out = out(reg) count,
        options(pure, nomem, nostack)
    );
    count
}

#[target_feature(enable = "sve2")]
unsafe fn filter_vec_sve_aux(
    input: *const u32,
    range: RangeInclusive<u32>,
    output: *mut u32,
    offset: u32,
    num_words: usize,
    vl: usize,
) -> usize {
    let range_start = *range.start();
    let range_end = *range.end();
    let highest_bit = HIGHEST_BIT;

    let mut output_cursor: usize = 0;
    let mut input_offset: usize = 0;
    let mut current_idx = offset;

    for _ in 0..num_words {
        let input_ptr = input.add(input_offset);
        let output_ptr = output.add(output_cursor);

        let added: usize;
        asm!(
            // Create all-true predicate for 32-bit elements
            "ptrue p0.s",

            // Load input values: z0 = input[input_offset..input_offset+vl]
            "ld1w {{ z0.s }}, p0/z, [{input_ptr}]",

            // Create index vector starting at current_idx: z1 = [current_idx, current_idx+1, ...]
            "index z1.s, {current_idx:w}, #1",

            // Transform u32 to i32 by XORing with highest bit for proper signed comparison
            // z2 = z0 ^ highest_bit (transformed values)
            "mov z3.s, {highest_bit:w}",
            "eor z2.s, z0.s, z3.s",

            // Compare: p1 = (z2 >= range_start)
            "mov z4.s, {range_start:w}",
            "cmpge p1.s, p0/z, z2.s, z4.s",

            // Compare: p2 = (z2 <= range_end)
            "mov z5.s, {range_end:w}",
            "cmple p2.s, p0/z, z2.s, z5.s",

            // p3 = p1 AND p2 (elements in range)
            "and p3.b, p0/z, p1.b, p2.b",

            // Compact: move indices where predicate is true to front of z6
            "compact z6.s, p3, z1.s",

            // Store compacted indices
            "st1w {{ z6.s }}, p0, [{output_ptr}]",

            // Count number of true predicates
            "cntp {added}, p0, p3.s",

            input_ptr = in(reg) input_ptr,
            output_ptr = in(reg) output_ptr,
            current_idx = in(reg) current_idx,
            highest_bit = in(reg) highest_bit,
            range_start = in(reg) range_start,
            range_end = in(reg) range_end,
            added = out(reg) added,
            out("p0") _,
            out("p1") _,
            out("p2") _,
            out("p3") _,
            out("z0") _,
            out("z1") _,
            out("z2") _,
            out("z3") _,
            out("z4") _,
            out("z5") _,
            out("z6") _,
            options(nostack)
        );

        output_cursor += added;
        input_offset += vl;
        current_idx += vl as u32;
    }

    output_cursor
}
