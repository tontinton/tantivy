tantivy is an embeddable log search engine library (similar to Lucene)

- NO TRIVIAL COMMENTS
- Because this is a fork and we want rebase with few conflicts, try to not modify existing code as much as possible, only adding new code
- Follow Rust idioms and best practices
- Latest Rust features can be used
- Descriptive variable and function names
- No wildcard imports
- Explicit error handling with `Result<T, E>` over panics
- Use custom error types using `TantivyError` in `src/error.rs`
- Format: `cargo +nightly fmt`
- Lint: `cargo clippy --tests --all-features`
- Place unit tests in the same file using `#[cfg(test)]` modules
- Run tests with: `cargo nextest run --features=mmap,quickwit,failpoints`
- Add dependencies to `Cargo.toml`
- Prefer well-maintained crates from crates.io
- Be mindful of allocations in hot paths
- Prefer structured logging
- Provide helpful error messages

