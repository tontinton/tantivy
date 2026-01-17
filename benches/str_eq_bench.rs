use binggan::plugins::PeakMemAllocPlugin;
use binggan::{black_box, InputGroup, PeakMemAlloc, INSTRUMENTED_SYSTEM};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::Count;
use tantivy::query::{FastFieldEqualityQuery, PhraseQuery, TermQuery};
use tantivy::schema::{IndexRecordOption, Schema, FAST, STRING, TEXT};
use tantivy::{doc, Index, Term};

#[global_allocator]
pub static GLOBAL: &PeakMemAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

fn main() {
    // Benchmark 1: Single token exact match (STRING field)
    // TermQuery on STRING field = exact match (entire field is one token)
    println!("=== Single Token Exact Match (STRING | FAST) ===");
    println!("TermQuery on STRING = exact match (no tokenization)\n");

    let string_inputs = vec![
        ("1M_10_terms", get_string_index(1_000_000, 10).unwrap()),
        ("1M_1K_terms", get_string_index(1_000_000, 1_000).unwrap()),
        (
            "1M_100K_terms",
            get_string_index(1_000_000, 100_000).unwrap(),
        ),
    ];
    bench_single_token(InputGroup::new_with_inputs(string_inputs));

    // Benchmark 2: Single token with TEXT field (tokenized but single word)
    // Comparing FastFieldStrEq vs PhraseQuery+match_entire on single token
    println!("\n=== Single Token on TEXT | FAST (tokenized field, single word) ===");
    println!("PhraseQuery requires 2+ terms, so we use a 2-word phrase for fair comparison\n");

    let text_single_inputs = vec![
        (
            "1M_10_terms",
            get_text_single_token_index(1_000_000, 10).unwrap(),
        ),
        (
            "1M_1K_terms",
            get_text_single_token_index(1_000_000, 1_000).unwrap(),
        ),
        (
            "1M_100K_terms",
            get_text_single_token_index(1_000_000, 100_000).unwrap(),
        ),
    ];
    bench_single_token_text(InputGroup::new_with_inputs(text_single_inputs));

    // Benchmark 3: Multi-word phrase exact match (TEXT field)
    // PhraseQuery with match_entire_field on TEXT = exact phrase match
    println!("\n=== Multi-Word Phrase Exact Match (TEXT | FAST) ===");
    println!("PhraseQuery+match_entire_field on TEXT = exact phrase match\n");

    let text_inputs = vec![
        ("1M_10_phrases", get_text_index(1_000_000, 10).unwrap()),
        ("1M_1K_phrases", get_text_index(1_000_000, 1_000).unwrap()),
        (
            "1M_100K_phrases",
            get_text_index(1_000_000, 100_000).unwrap(),
        ),
    ];
    bench_phrase(InputGroup::new_with_inputs(text_inputs));
}

fn bench_single_token(mut group: InputGroup<Index>) {
    group.add_plugin(PeakMemAllocPlugin::new(GLOBAL));

    group.register("fast_field_str_eq", |index| {
        fast_field_str_eq_query(index, "event", "term_0");
    });

    group.register("term_query", |index| {
        term_query(index, "event", "term_0");
    });

    group.register("fast_field_nonexistent", |index| {
        fast_field_str_eq_query(index, "event", "nonexistent");
    });

    group.register("term_query_nonexistent", |index| {
        term_query(index, "event", "nonexistent");
    });

    group.run();
}

fn bench_single_token_text(mut group: InputGroup<Index>) {
    group.add_plugin(PeakMemAllocPlugin::new(GLOBAL));

    // Using 2-word values like "term 0" since PhraseQuery requires 2+ terms
    group.register("fast_field_str_eq", |index| {
        fast_field_str_eq_query(index, "event", "term 0");
    });

    group.register("phrase_match_entire", |index| {
        phrase_match_entire_field(index, "event", "term 0");
    });

    group.register("fast_field_nonexistent", |index| {
        fast_field_str_eq_query(index, "event", "nonexistent value");
    });

    group.register("phrase_nonexistent", |index| {
        phrase_match_entire_field(index, "event", "nonexistent value");
    });

    group.run();
}

fn bench_phrase(mut group: InputGroup<Index>) {
    group.add_plugin(PeakMemAllocPlugin::new(GLOBAL));

    group.register("fast_field_str_eq", |index| {
        fast_field_str_eq_query(index, "event", "hello world 0");
    });

    group.register("phrase_match_entire", |index| {
        phrase_match_entire_field(index, "event", "hello world 0");
    });

    group.register("fast_field_nonexistent", |index| {
        fast_field_str_eq_query(index, "event", "nonexistent phrase here");
    });

    group.register("phrase_nonexistent", |index| {
        phrase_match_entire_field(index, "event", "nonexistent phrase here");
    });

    group.run();
}

fn fast_field_str_eq_query(index: &Index, field_name: &str, value: &str) {
    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let field = searcher.schema().get_field(field_name).unwrap();

    let query = FastFieldEqualityQuery::new(field, value);
    let count = searcher.search(&query, &Count).unwrap();
    black_box(count);
}

fn term_query(index: &Index, field_name: &str, value: &str) {
    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let field = searcher.schema().get_field(field_name).unwrap();

    let term = Term::from_field_text(field, value);
    let query = TermQuery::new(term, IndexRecordOption::Basic);
    let count = searcher.search(&query, &Count).unwrap();
    black_box(count);
}

fn phrase_match_entire_field(index: &Index, field_name: &str, value: &str) {
    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let field = searcher.schema().get_field(field_name).unwrap();

    let terms: Vec<Term> = value
        .split_whitespace()
        .map(|word| Term::from_field_text(field, word))
        .collect();

    let mut query = PhraseQuery::new(terms);
    query.set_match_entire_field(true);

    let count = searcher.search(&query, &Count).unwrap();
    black_box(count);
}

/// Creates index with STRING | FAST field (single token, no tokenization)
fn get_string_index(num_docs: usize, num_unique_terms: usize) -> tantivy::Result<Index> {
    let mut schema_builder = Schema::builder();
    let event_field = schema_builder.add_text_field("event", STRING | FAST);
    let schema = schema_builder.build();

    let index = Index::create_from_tempdir(schema)?;

    let terms: Vec<String> = (0..num_unique_terms)
        .map(|i| format!("term_{}", i))
        .collect();

    let mut rng = StdRng::from_seed([42u8; 32]);
    let mut index_writer = index.writer_with_num_threads(1, 100_000_000)?;

    for _ in 0..num_docs {
        let term = terms.choose(&mut rng).unwrap();
        index_writer.add_document(doc!(event_field => term.as_str()))?;
    }

    index_writer.commit()?;
    Ok(index)
}

/// Creates index with TEXT | FAST field (tokenized into words)
fn get_text_index(num_docs: usize, num_unique_phrases: usize) -> tantivy::Result<Index> {
    let mut schema_builder = Schema::builder();
    let event_field = schema_builder.add_text_field("event", TEXT | FAST);
    let schema = schema_builder.build();

    let index = Index::create_from_tempdir(schema)?;

    let phrases: Vec<String> = (0..num_unique_phrases)
        .map(|i| format!("hello world {}", i))
        .collect();

    let mut rng = StdRng::from_seed([42u8; 32]);
    let mut index_writer = index.writer_with_num_threads(1, 100_000_000)?;

    for _ in 0..num_docs {
        let phrase = phrases.choose(&mut rng).unwrap();
        index_writer.add_document(doc!(event_field => phrase.as_str()))?;
    }

    index_writer.commit()?;
    Ok(index)
}

/// Creates index with TEXT | FAST field with short 2-word values (for PhraseQuery comparison)
fn get_text_single_token_index(num_docs: usize, num_unique_terms: usize) -> tantivy::Result<Index> {
    let mut schema_builder = Schema::builder();
    let event_field = schema_builder.add_text_field("event", TEXT | FAST);
    let schema = schema_builder.build();

    let index = Index::create_from_tempdir(schema)?;

    // Using 2-word phrases like "term 0", "term 1" since PhraseQuery requires 2+ terms
    let terms: Vec<String> = (0..num_unique_terms)
        .map(|i| format!("term {}", i))
        .collect();

    let mut rng = StdRng::from_seed([42u8; 32]);
    let mut index_writer = index.writer_with_num_threads(1, 100_000_000)?;

    for _ in 0..num_docs {
        let term = terms.choose(&mut rng).unwrap();
        index_writer.add_document(doc!(event_field => term.as_str()))?;
    }

    index_writer.commit()?;
    Ok(index)
}
