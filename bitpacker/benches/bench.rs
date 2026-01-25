#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use rand::seq::IteratorRandom;
    use rand::{thread_rng, Rng, SeedableRng};
    use tantivy_bitpacker::{BitPacker, BitUnpacker, BlockedBitpacker};
    use test::Bencher;

    #[inline(never)]
    fn create_bitpacked_data(bit_width: u8, num_els: u32) -> Vec<u8> {
        let mut bitpacker = BitPacker::new();
        let mut buffer = Vec::new();
        for _ in 0..num_els {
            // the values do not matter.
            bitpacker.write(0u64, bit_width, &mut buffer).unwrap();
            bitpacker.flush(&mut buffer).unwrap();
        }
        buffer
    }

    #[bench]
    fn bench_bitpacking_read(b: &mut Bencher) {
        let bit_width = 3;
        let num_els = 1_000_000u32;
        let bit_unpacker = BitUnpacker::new(bit_width);
        let data = create_bitpacked_data(bit_width, num_els);
        let idxs: Vec<u32> = (0..num_els).choose_multiple(&mut thread_rng(), 100_000);
        b.iter(|| {
            let mut out = 0u64;
            for &idx in &idxs {
                out = out.wrapping_add(bit_unpacker.get(idx, &data[..]));
            }
            out
        });
    }

    #[bench]
    fn bench_blockedbitp_read(b: &mut Bencher) {
        let mut blocked_bitpacker = BlockedBitpacker::new();
        for val in 0..=21500 {
            blocked_bitpacker.add(val * val);
        }
        b.iter(|| {
            let mut out = 0u64;
            for val in 0..=21500 {
                out = out.wrapping_add(blocked_bitpacker.get(val));
            }
            out
        });
    }

    #[bench]
    fn bench_blockedbitp_create(b: &mut Bencher) {
        b.iter(|| {
            let mut blocked_bitpacker = BlockedBitpacker::new();
            for val in 0..=21500 {
                blocked_bitpacker.add(val * val);
            }
            blocked_bitpacker
        });
    }

    fn create_filter_data(num_els: usize) -> Vec<u32> {
        let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
        (0..num_els).map(|_| rng.gen_range(0..128u32)).collect()
    }

    #[bench]
    fn bench_filter_vec_scalar(b: &mut Bencher) {
        let data = create_filter_data(100_000);
        let range = 0u32..=63u32;
        b.iter(|| {
            let mut output = data.clone();
            tantivy_bitpacker::filter_vec_in_place_scalar(range.clone(), 0, &mut output);
            output
        });
    }

    #[bench]
    fn bench_filter_vec_avx2(b: &mut Bencher) {
        let data = create_filter_data(100_000);
        let range = 0u32..=63u32;
        b.iter(|| {
            let mut output = data.clone();
            tantivy_bitpacker::filter_vec_in_place_avx2(range.clone(), 0, &mut output);
            output
        });
    }

    #[bench]
    fn bench_filter_vec_avx512(b: &mut Bencher) {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let data = create_filter_data(100_000);
        let range = 0u32..=63u32;
        b.iter(|| {
            let mut output = data.clone();
            tantivy_bitpacker::filter_vec_in_place_avx512(range.clone(), 0, &mut output);
            output
        });
    }
}
