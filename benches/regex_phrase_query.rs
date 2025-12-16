use binggan::black_box;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;
use tantivy::collector::Count;
use tantivy::query::RegexPhraseQuery;
use tantivy::schema::{Schema, TEXT};
use tantivy::{doc, Index};

const DOCS: usize = 10_000_000;
const ITERS: usize = 10;

fn build_and_save_index(path: &Path) -> Index {
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_dir(path, schema).unwrap();

    {
        let mut writer = index.writer_with_num_threads(1, 500_000_000).unwrap();
        let mut rng = StdRng::from_seed([42u8; 32]);

        for _ in 0..DOCS {
            let prefix_words: Vec<&str> = vec!["small", "tiny", "big", "large", "huge"];
            let infix_words: Vec<&str> = vec!["red", "brown", "scary", "dangerous", "wild"];
            let suffix_words: Vec<&str> = vec!["dog", "cat", "wolf", "bear", "lion"];

            let prefix = prefix_words[rng.gen_range(0..prefix_words.len())];
            let infix = infix_words[rng.gen_range(0..infix_words.len())];
            let suffix = suffix_words[rng.gen_range(0..suffix_words.len())];

            // Create document text with varying distances between words
            let text = match rng.gen_range(0..3) {
                0 => format!("{} {}", prefix, suffix),
                1 => format!("{} {} {}", prefix, infix, suffix),
                _ => format!("{} {} {} {}", prefix, infix, infix, suffix),
            };

            writer
                .add_document(doc!(text_field => text))
                .expect("failed to add document");
        }
        writer.commit().unwrap();
    }

    index
}

fn load_or_build_index(path: &Path, force_build: bool) -> Index {
    if !force_build && path.exists() {
        Index::open_in_dir(path).unwrap_or_else(|_| {
            eprintln!("Failed to open index at {:?}, rebuilding...", path);
            build_and_save_index(path)
        })
    } else {
        if path.exists() {
            std::fs::remove_dir_all(path).expect("Failed to remove existing index directory");
        }
        std::fs::create_dir_all(path).expect("Failed to create index directory");
        build_and_save_index(path)
    }
}

fn main() {
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let mut index_path = "./regex_phrase_index".to_string();
    let mut force_build = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--index" => {
                i += 1;
                if i < args.len() {
                    index_path = args[i].clone();
                }
            }
            "--build" => {
                force_build = true;
            }
            "--help" => {
                println!("Usage: regex_phrase_query [OPTIONS]");
                println!("  --index PATH    Directory to load/save index (default: ./regex_phrase_index)");
                println!("  --build         Force rebuild the index");
                println!("  --help          Show this help message");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let start_total = Instant::now();

    let start_index = Instant::now();
    let index = load_or_build_index(Path::new(&index_path), force_build);
    let index_time = start_index.elapsed();
    println!(
        "Index load/build time: {:.2}ms",
        index_time.as_secs_f64() * 1000.0
    );

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();

    // Test 1: Simple regex (few term matches)
    let mut query = RegexPhraseQuery::new(
        searcher.schema().get_field("text").unwrap(),
        vec!["larg.*".to_string(), "wolf".to_string()],
    );
    query.set_slop(0);

    println!("Query: 'larg.* wolf' (few term matches)");

    let start_query = Instant::now();
    let mut total_results = 0u64;

    for _ in 0..ITERS {
        let count = searcher.search(&query, &Count).unwrap();
        total_results += count as u64;
        black_box(count);
    }
    let query_time = start_query.elapsed();
    println!(
        "slop=0 Total query time: {:.2}ms",
        query_time.as_secs_f64() * 1000.0
    );
    println!(
        "slop=0 Average query time: {:.2}ms",
        query_time.as_secs_f64() * 1000.0 / ITERS as f64
    );

    // Test with high slop (unlimited slop optimization)
    let mut query_high_slop = RegexPhraseQuery::new(
        searcher.schema().get_field("text").unwrap(),
        vec!["larg.*".to_string(), "wolf".to_string()],
    );
    query_high_slop.set_slop(u32::MAX);

    let start_query_high = Instant::now();
    let mut total_results_high = 0u64;

    for _ in 0..ITERS {
        let count = searcher.search(&query_high_slop, &Count).unwrap();
        total_results_high += count as u64;
        black_box(count);
    }
    let query_time_high = start_query_high.elapsed();
    println!(
        "slop=unlimited Total query time: {:.2}ms",
        query_time_high.as_secs_f64() * 1000.0
    );
    println!(
        "slop=unlimited Average query time: {:.2}ms",
        query_time_high.as_secs_f64() * 1000.0 / ITERS as f64
    );

    // Test 3: Regex matching many terms (.*a.* matches all words with 'a')
    let mut query_many = RegexPhraseQuery::new(
        searcher.schema().get_field("text").unwrap(),
        vec![".*a.*".to_string(), ".*o.*".to_string()],
    );
    query_many.set_slop(0);

    println!("\nQuery: '.*a.* .*o.*' (many term matches)");
    let start_query_many = Instant::now();
    let mut total_results_many = 0u64;

    for _ in 0..1 {
        let count = searcher.search(&query_many, &Count).unwrap();
        total_results_many += count as u64;
        black_box(count);
    }
    let query_time_many = start_query_many.elapsed();
    println!(
        "slop=0 Total query time: {:.2}ms",
        query_time_many.as_secs_f64() * 1000.0
    );
    // println!(
    //     "slop=0 Average query time: {:.2}ms",
    //     query_time_many.as_secs_f64() * 1000.0 / ITERS as f64
    // );

    // Test with high slop on many-term query
    query_many.set_slop(u32::MAX);
    let start_query_many_high = Instant::now();
    let mut total_results_many_high = 0u64;

    for _ in 0..1 {
        let count = searcher.search(&query_many, &Count).unwrap();
        total_results_many_high += count as u64;
        black_box(count);
    }
    let query_time_many_high = start_query_many_high.elapsed();
    println!(
        "slop=unlimited Total query time: {:.2}ms",
        query_time_many_high.as_secs_f64() * 1000.0
    );
    // println!(
    //     "slop=5000 Average query time: {:.2}ms",
    //     query_time_many_high.as_secs_f64() * 1000.0 / ITERS as f64
    // );

    println!(
        "\nTotal time: {:.2}ms",
        start_total.elapsed().as_secs_f64() * 1000.0
    );
    black_box(total_results);
    black_box(total_results_high);
    black_box(total_results_many);
    black_box(total_results_many_high);
}
