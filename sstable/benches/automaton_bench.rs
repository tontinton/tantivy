use std::collections::BTreeSet;

use common::OwnedBytes;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tantivy_fst::Automaton;
use tantivy_sstable::{SSTableIndex, SSTableIndexBuilder, SSTableIndexV3};

const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

fn generate_key(rng: &mut impl Rng) -> Vec<u8> {
    let len = rng.gen_range(3..15);
    (0..len)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx]
        })
        .collect()
}

fn prepare_large_sstable(num_keys: usize) -> Result<SSTableIndex, Box<dyn std::error::Error>> {
    let mut rng = StdRng::from_seed([42u8; 32]);
    let mut keys = BTreeSet::new();

    while keys.len() < num_keys {
        keys.insert(generate_key(&mut rng));
    }

    let mut builder = SSTableIndexBuilder::default();
    for (ord, key) in keys.iter().enumerate() {
        let start = ord * 100;
        let end = start + 100;
        builder.add_block(key, start..end, ord as u64, None);
    }

    let mut buffer = Vec::new();
    let fst_len = builder.serialize(&mut buffer)?;
    let buffer = OwnedBytes::new(buffer);
    let v3_index =
        SSTableIndexV3::load(buffer, fst_len).map_err(|_| "Failed to load SSTableIndexV3")?;
    Ok(SSTableIndex::V3(v3_index))
}

// Helper automaton for prefix matching (matches "prefix.*")
#[derive(Clone)]
struct PrefixAutomaton {
    prefix: Vec<u8>,
}

impl PrefixAutomaton {
    fn new(prefix: &[u8]) -> Self {
        Self {
            prefix: prefix.to_vec(),
        }
    }
}

impl Automaton for PrefixAutomaton {
    type State = usize; // Position in prefix, or prefix.len() if matched, usize::MAX if dead

    fn start(&self) -> Self::State {
        0
    }

    fn is_match(&self, &state: &Self::State) -> bool {
        state == self.prefix.len()
    }

    fn can_match(&self, &state: &Self::State) -> bool {
        state <= self.prefix.len()
    }

    fn will_always_match(&self, &state: &Self::State) -> bool {
        state == self.prefix.len()
    }

    fn accept(&self, &state: &Self::State, byte: u8) -> Self::State {
        if state == self.prefix.len() {
            // Already matched prefix, stay in match state
            self.prefix.len()
        } else if state < self.prefix.len() && self.prefix[state] == byte {
            // Still matching prefix
            state + 1
        } else {
            // Mismatch - dead state (usize::MAX will be > prefix.len())
            usize::MAX
        }
    }
}

fn bench_automaton_traversal<A>(
    c: &mut Criterion,
    name: &str,
    sstable: &SSTableIndex,
    automaton: A,
) where
    A: Automaton + Clone + 'static,
{
    c.bench_function(name, |b| {
        b.iter(|| {
            let count = sstable
                .get_block_for_automaton(black_box(&automaton))
                .count();
            black_box(count)
        })
    });
}

pub fn criterion_benchmark(c: &mut Criterion) {
    eprintln!("Building large SSTable with 100,000 keys...");
    let sstable = prepare_large_sstable(100_000).expect("Failed to build SSTable");
    eprintln!("SSTable built successfully!");

    // Benchmark 1: Very selective prefix (should match ~0.4% of keys)
    eprintln!("\nBenchmarking: Highly selective prefix 'aa'");
    let prefix_aa = PrefixAutomaton::new(b"aa");
    let count_aa = sstable.get_block_for_automaton(&prefix_aa).count();
    eprintln!("  Expected matches: ~400, Actual: {}", count_aa);
    bench_automaton_traversal(c, "prefix_highly_selective_aa", &sstable, prefix_aa);

    // Benchmark 2: Selective prefix (should match ~4% of keys)
    eprintln!("\nBenchmarking: Selective prefix 'a'");
    let prefix_a = PrefixAutomaton::new(b"a");
    let count_a = sstable.get_block_for_automaton(&prefix_a).count();
    eprintln!("  Expected matches: ~4000, Actual: {}", count_a);
    bench_automaton_traversal(c, "prefix_selective_a", &sstable, prefix_a);

    // Benchmark 3: Medium selectivity prefix
    eprintln!("\nBenchmarking: Medium selectivity prefix 'm'");
    let prefix_m = PrefixAutomaton::new(b"m");
    let count_m = sstable.get_block_for_automaton(&prefix_m).count();
    eprintln!("  Expected matches: ~4000, Actual: {}", count_m);
    bench_automaton_traversal(c, "prefix_medium_m", &sstable, prefix_m);

    // Benchmark 4: Multi-byte prefix
    eprintln!("\nBenchmarking: Multi-byte prefix 'test'");
    let prefix_test = PrefixAutomaton::new(b"test");
    let count_test = sstable.get_block_for_automaton(&prefix_test).count();
    eprintln!("  Expected matches: ~10-100, Actual: {}", count_test);
    bench_automaton_traversal(c, "prefix_multibyte_test", &sstable, prefix_test);

    // Benchmark 5: Empty prefix (matches everything - baseline)
    eprintln!("\nBenchmarking: Empty prefix (full scan baseline)");
    let prefix_empty = PrefixAutomaton::new(b"");
    let count_empty = sstable.get_block_for_automaton(&prefix_empty).count();
    eprintln!("  Expected matches: 100000, Actual: {}", count_empty);
    bench_automaton_traversal(c, "prefix_empty_full_scan", &sstable, prefix_empty);

    // Benchmark 6: Union of two prefixes
    eprintln!("\nBenchmarking: Union of prefixes 'aa' | 'bb'");
    let prefix_aa = PrefixAutomaton::new(b"aa");
    let prefix_bb = PrefixAutomaton::new(b"bb");
    let union = prefix_aa.union(prefix_bb);
    let count_union = sstable.get_block_for_automaton(&union).count();
    eprintln!("  Expected matches: ~800, Actual: {}", count_union);
    bench_automaton_traversal(c, "prefix_union_aa_bb", &sstable, union);

    // Benchmark 7: Non-matching prefix (should match 0 keys)
    eprintln!("\nBenchmarking: Non-matching prefix '999'");
    let prefix_999 = PrefixAutomaton::new(b"999");
    let count_999 = sstable.get_block_for_automaton(&prefix_999).count();
    eprintln!("  Expected matches: 0, Actual: {}", count_999);
    bench_automaton_traversal(c, "prefix_no_match_999", &sstable, prefix_999);

    // Benchmark 8: High-value prefix (should benefit most from range optimization)
    eprintln!("\nBenchmarking: High-value prefix 'z'");
    let prefix_z = PrefixAutomaton::new(b"z");
    let count_z = sstable.get_block_for_automaton(&prefix_z).count();
    eprintln!("  Expected matches: ~4000, Actual: {}", count_z);
    bench_automaton_traversal(c, "prefix_high_value_z", &sstable, prefix_z);

    eprintln!("\n=== Benchmarks complete! ===\n");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
