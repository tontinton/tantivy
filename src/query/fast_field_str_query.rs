use common::BitSet;
use regex::Regex;

use crate::query::explanation::does_not_match;
use crate::query::fast_field_str_common::{
    collect_matching_ords, get_str_column_for_term, matching_ords_to_scorer,
};
use crate::query::{EmptyScorer, EnableScoring, Explanation, Query, Scorer, Weight};
use crate::schema::Term;
use crate::{DocId, Score, SegmentReader, TantivyError};

/// Specifies how to match strings in a `FastFieldStrQuery`.
#[derive(Debug, Clone)]
pub enum FastFieldStrMatchType {
    /// Match if term contains the pattern.
    Contains(String),
    /// Match if term starts with the prefix.
    StartsWith(String),
    /// Match if term ends with the suffix.
    EndsWith(String),
    /// Match against a compiled regex.
    Regex(Regex),
}

/// A unified query for string matching on fast fields using various match types.
///
/// `FastFieldStrQuery` matches documents where a string fast field matches according to
/// the specified `FastFieldStrMatchType`. It supports:
/// - `Contains`: substring matching
/// - `StartsWith`: prefix matching (with dictionary optimization for case-sensitive)
/// - `EndsWith`: suffix matching
/// - `Regex`: regular expression matching
///
/// This query iterates through unique terms in the dictionary and checks each against
/// the match predicate. It supports both regular string fields and JSON fields with paths.
///
/// For case-insensitive matching (except `Regex`, which uses regex flags), set
/// `case_insensitive` to `true`.
#[derive(Debug, Clone)]
pub struct FastFieldStrQuery {
    term: Term,
    match_type: FastFieldStrMatchType,
    case_insensitive: bool,
}

impl FastFieldStrQuery {
    /// Creates a new `FastFieldStrQuery`.
    ///
    /// # Arguments
    ///
    /// * `term` - The term containing the field (and optional JSON path). The term value is
    ///   ignored; only the field and path information is used.
    /// * `match_type` - The type of string matching to perform.
    pub fn new(term: Term, match_type: FastFieldStrMatchType) -> Self {
        Self {
            term,
            match_type,
            case_insensitive: false,
        }
    }

    /// Creates a `FastFieldStrQuery` for substring matching.
    ///
    /// # Arguments
    ///
    /// * `term` - The term containing the field (and optional JSON path).
    /// * `pattern` - The substring to search for.
    pub fn contains(term: Term, pattern: String) -> Self {
        Self::new(term, FastFieldStrMatchType::Contains(pattern))
    }

    /// Creates a `FastFieldStrQuery` for prefix matching.
    ///
    /// # Arguments
    ///
    /// * `term` - The term containing the field (and optional JSON path).
    /// * `prefix` - The prefix to match.
    pub fn starts_with(term: Term, prefix: String) -> Self {
        Self::new(term, FastFieldStrMatchType::StartsWith(prefix))
    }

    /// Creates a `FastFieldStrQuery` for suffix matching.
    ///
    /// # Arguments
    ///
    /// * `term` - The term containing the field (and optional JSON path).
    /// * `suffix` - The suffix to match.
    pub fn ends_with(term: Term, suffix: String) -> Self {
        Self::new(term, FastFieldStrMatchType::EndsWith(suffix))
    }

    /// Creates a `FastFieldStrQuery` for regex matching.
    ///
    /// Note: Case sensitivity is controlled by regex flags (e.g., `(?i)`), not by
    /// `set_case_insensitive`.
    ///
    /// # Arguments
    ///
    /// * `term` - The term containing the field (and optional JSON path).
    /// * `regex` - The compiled regex pattern.
    pub fn regex(term: Term, regex: Regex) -> Self {
        Self::new(term, FastFieldStrMatchType::Regex(regex))
    }

    /// Sets whether the search should be case-insensitive.
    ///
    /// Note: This has no effect on `Regex` match type, which uses regex flags instead.
    pub fn set_case_insensitive(&mut self, case_insensitive: bool) -> &mut Self {
        self.case_insensitive = case_insensitive;
        self
    }

    /// Returns whether this query is case-insensitive.
    pub fn is_case_insensitive(&self) -> bool {
        self.case_insensitive
    }

    /// Returns the match type of this query.
    pub fn match_type(&self) -> &FastFieldStrMatchType {
        &self.match_type
    }

    /// Returns the term associated with this query.
    pub fn term(&self) -> &Term {
        &self.term
    }
}

impl Query for FastFieldStrQuery {
    fn weight(&self, _enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(FastFieldStrWeight::new(
            self.term.clone(),
            self.match_type.clone(),
            self.case_insensitive,
        )))
    }
}

/// Weight for `FastFieldStrQuery`.
#[derive(Debug, Clone)]
pub struct FastFieldStrWeight {
    term: Term,
    match_type: FastFieldStrMatchType,
    case_insensitive: bool,
}

impl FastFieldStrWeight {
    /// Creates a new `FastFieldStrWeight`.
    pub fn new(term: Term, match_type: FastFieldStrMatchType, case_insensitive: bool) -> Self {
        Self {
            term,
            match_type,
            case_insensitive,
        }
    }

    fn query_name(&self) -> &'static str {
        match &self.match_type {
            FastFieldStrMatchType::Contains(_) => "fast field str contains query",
            FastFieldStrMatchType::StartsWith(_) => "fast field str starts_with query",
            FastFieldStrMatchType::EndsWith(_) => "fast field str ends_with query",
            FastFieldStrMatchType::Regex(_) => "fast field str regex query",
        }
    }
}

impl Weight for FastFieldStrWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let Some((str_column, field_name)) =
            get_str_column_for_term(reader, &self.term, self.query_name())?
        else {
            return Ok(Box::new(EmptyScorer));
        };

        let matching_ords = match &self.match_type {
            FastFieldStrMatchType::Contains(pattern) => {
                let pattern_for_match = if self.case_insensitive {
                    pattern.to_lowercase()
                } else {
                    pattern.clone()
                };
                let case_insensitive = self.case_insensitive;
                collect_matching_ords(&str_column, &field_name, |term_text| {
                    if case_insensitive {
                        term_text.to_lowercase().contains(&pattern_for_match)
                    } else {
                        term_text.contains(&pattern_for_match)
                    }
                })?
            }
            FastFieldStrMatchType::StartsWith(prefix) => {
                if self.case_insensitive {
                    let prefix_lower = prefix.to_lowercase();
                    collect_matching_ords(&str_column, &field_name, |term_text| {
                        term_text.to_lowercase().starts_with(&prefix_lower)
                    })?
                } else {
                    let mut matching_ords = BitSet::with_max_value(str_column.num_terms() as u32);
                    let dictionary = str_column.dictionary();
                    let mut stream = dictionary
                        .prefix_range(prefix.as_bytes())
                        .into_stream()
                        .map_err(|e| {
                            TantivyError::InternalError(format!(
                                "failed to create prefix stream for {field_name}: {e:?}"
                            ))
                        })?;

                    while stream.advance() {
                        matching_ords.insert(stream.term_ord() as u32);
                    }
                    matching_ords
                }
            }
            FastFieldStrMatchType::EndsWith(suffix) => {
                let suffix_for_match = if self.case_insensitive {
                    suffix.to_lowercase()
                } else {
                    suffix.clone()
                };
                let case_insensitive = self.case_insensitive;
                collect_matching_ords(&str_column, &field_name, |term_text| {
                    if case_insensitive {
                        term_text.to_lowercase().ends_with(&suffix_for_match)
                    } else {
                        term_text.ends_with(&suffix_for_match)
                    }
                })?
            }
            FastFieldStrMatchType::Regex(regex) => {
                collect_matching_ords(&str_column, &field_name, |term_text| {
                    regex.is_match(term_text)
                })?
            }
        };

        Ok(matching_ords_to_scorer(
            &str_column,
            &matching_ords,
            reader.max_doc(),
            boost,
        ))
    }

    fn explain(&self, _reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        Err(does_not_match(doc))
    }
}

#[cfg(test)]
mod tests {
    use regex::Regex;
    use serde_json::json;

    use crate::collector::{Count, TopDocs};
    use crate::query::{FastFieldStrMatchType, FastFieldStrQuery};
    use crate::schema::{Schema, FAST, STRING, TEXT};
    use crate::{Index, IndexWriter, Term};

    #[test]
    fn test_contains_basic() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", TEXT | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "hello world"))?;
        index_writer.add_document(doc!(title_field => "goodbye world"))?;
        index_writer.add_document(doc!(title_field => "hello there"))?;
        index_writer.add_document(doc!(title_field => "nothing"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::contains(
            Term::from_field_text(title_field, ""),
            "hello".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query = FastFieldStrQuery::contains(
            Term::from_field_text(title_field, ""),
            "world".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query =
            FastFieldStrQuery::contains(Term::from_field_text(title_field, ""), "xyz".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_contains_case_insensitive() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", STRING | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "Hello World"))?;
        index_writer.add_document(doc!(title_field => "HELLO WORLD"))?;
        index_writer.add_document(doc!(title_field => "hello world"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::contains(
            Term::from_field_text(title_field, ""),
            "Hello".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let mut query = FastFieldStrQuery::contains(
            Term::from_field_text(title_field, ""),
            "Hello".to_string(),
        );
        query.set_case_insensitive(true);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);

        Ok(())
    }

    #[test]
    fn test_starts_with_basic() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", TEXT | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "hello world"))?;
        index_writer.add_document(doc!(title_field => "hello there"))?;
        index_writer.add_document(doc!(title_field => "goodbye world"))?;
        index_writer.add_document(doc!(title_field => "hi there"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::starts_with(
            Term::from_field_text(title_field, ""),
            "hello".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query =
            FastFieldStrQuery::starts_with(Term::from_field_text(title_field, ""), "h".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);

        let query = FastFieldStrQuery::starts_with(
            Term::from_field_text(title_field, ""),
            "xyz".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_starts_with_case_insensitive() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", STRING | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "Hello World"))?;
        index_writer.add_document(doc!(title_field => "HELLO there"))?;
        index_writer.add_document(doc!(title_field => "hello world"))?;
        index_writer.add_document(doc!(title_field => "Goodbye"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::starts_with(
            Term::from_field_text(title_field, ""),
            "Hello".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let mut query = FastFieldStrQuery::starts_with(
            Term::from_field_text(title_field, ""),
            "Hello".to_string(),
        );
        query.set_case_insensitive(true);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);

        Ok(())
    }

    #[test]
    fn test_ends_with_basic() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", TEXT | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "hello world"))?;
        index_writer.add_document(doc!(title_field => "goodbye world"))?;
        index_writer.add_document(doc!(title_field => "hello there"))?;
        index_writer.add_document(doc!(title_field => "test"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::ends_with(
            Term::from_field_text(title_field, ""),
            "world".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query = FastFieldStrQuery::ends_with(
            Term::from_field_text(title_field, ""),
            "there".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let query =
            FastFieldStrQuery::ends_with(Term::from_field_text(title_field, ""), "xyz".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_ends_with_case_insensitive() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", STRING | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "hello World"))?;
        index_writer.add_document(doc!(title_field => "hello WORLD"))?;
        index_writer.add_document(doc!(title_field => "hello world"))?;
        index_writer.add_document(doc!(title_field => "goodbye"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::ends_with(
            Term::from_field_text(title_field, ""),
            "World".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let mut query = FastFieldStrQuery::ends_with(
            Term::from_field_text(title_field, ""),
            "World".to_string(),
        );
        query.set_case_insensitive(true);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);

        Ok(())
    }

    #[test]
    fn test_regex_basic() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", TEXT | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "cmd.exe"))?;
        index_writer.add_document(doc!(title_field => "powershell.exe"))?;
        index_writer.add_document(doc!(title_field => "notepad.exe"))?;
        index_writer.add_document(doc!(title_field => "script.sh"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::regex(
            Term::from_field_text(title_field, ""),
            Regex::new("^(cmd|powershell)\\.exe$").unwrap(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query = FastFieldStrQuery::regex(
            Term::from_field_text(title_field, ""),
            Regex::new("\\.exe$").unwrap(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);

        let query = FastFieldStrQuery::regex(
            Term::from_field_text(title_field, ""),
            Regex::new("(?i)CMD").unwrap(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        Ok(())
    }

    #[test]
    fn test_json_field_with_path() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer
            .add_document(doc!(json_field => json!({"user": {"command": "powershell.exe"}})))?;
        index_writer.add_document(doc!(json_field => json!({"user": {"command": "cmd.exe"}})))?;
        index_writer.add_document(
            doc!(json_field => json!({"user": {"command": "powershell.exe -enc"}})),
        )?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let mut command_term = Term::from_field_json_path(json_field, "user.command", false);
        command_term.append_type_and_str("");

        let query = FastFieldStrQuery::contains(command_term.clone(), "powershell".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query = FastFieldStrQuery::starts_with(command_term.clone(), "cmd".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let query = FastFieldStrQuery::ends_with(command_term.clone(), ".exe".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query =
            FastFieldStrQuery::regex(command_term.clone(), Regex::new("(?i)powershell").unwrap());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        Ok(())
    }

    #[test]
    fn test_multivalued_field() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let tags_field = schema_builder.add_text_field("tags", TEXT | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(tags_field => "warning", tags_field => "error"))?;
        index_writer.add_document(doc!(tags_field => "debug", tags_field => "info"))?;
        index_writer.add_document(doc!(tags_field => "warn"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query =
            FastFieldStrQuery::contains(Term::from_field_text(tags_field, ""), "warn".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query = FastFieldStrQuery::starts_with(
            Term::from_field_text(tags_field, ""),
            "warn".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        Ok(())
    }

    #[test]
    fn test_unicode() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", STRING | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "日本語テスト"))?;
        index_writer.add_document(doc!(title_field => "テスト"))?;
        index_writer.add_document(doc!(title_field => "hello日本語"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::contains(
            Term::from_field_text(title_field, ""),
            "テスト".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query = FastFieldStrQuery::starts_with(
            Term::from_field_text(title_field, ""),
            "日本".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let query = FastFieldStrQuery::ends_with(
            Term::from_field_text(title_field, ""),
            "日本語".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        Ok(())
    }

    #[test]
    fn test_empty_pattern() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", STRING | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "hello"))?;
        index_writer.add_document(doc!(title_field => "world"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query =
            FastFieldStrQuery::contains(Term::from_field_text(title_field, ""), "".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query =
            FastFieldStrQuery::starts_with(Term::from_field_text(title_field, ""), "".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query =
            FastFieldStrQuery::ends_with(Term::from_field_text(title_field, ""), "".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        Ok(())
    }

    #[test]
    fn test_empty_index() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", STRING | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let writer: IndexWriter = index.writer_for_tests()?;
        drop(writer);

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::contains(
            Term::from_field_text(title_field, ""),
            "hello".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        let query = FastFieldStrQuery::starts_with(
            Term::from_field_text(title_field, ""),
            "hello".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        let query = FastFieldStrQuery::ends_with(
            Term::from_field_text(title_field, ""),
            "hello".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        let query = FastFieldStrQuery::regex(
            Term::from_field_text(title_field, ""),
            Regex::new("hello").unwrap(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_multiple_segments() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", STRING | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "hello world"))?;
        index_writer.commit()?;

        index_writer.add_document(doc!(title_field => "hello there"))?;
        index_writer.commit()?;

        index_writer.add_document(doc!(title_field => "goodbye"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        assert_eq!(searcher.segment_readers().len(), 3);

        let query = FastFieldStrQuery::contains(
            Term::from_field_text(title_field, ""),
            "hello".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        Ok(())
    }

    #[test]
    fn test_with_top_docs() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", STRING | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "hello world"))?;
        index_writer.add_document(doc!(title_field => "goodbye"))?;
        index_writer.add_document(doc!(title_field => "hello there"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::contains(
            Term::from_field_text(title_field, ""),
            "hello".to_string(),
        );
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 2);

        Ok(())
    }

    #[test]
    fn test_match_type_accessor() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", STRING | FAST);
        let _schema = schema_builder.build();

        let query =
            FastFieldStrQuery::contains(Term::from_field_text(title_field, ""), "test".to_string());
        assert!(matches!(
            query.match_type(),
            FastFieldStrMatchType::Contains(_)
        ));

        let query = FastFieldStrQuery::starts_with(
            Term::from_field_text(title_field, ""),
            "test".to_string(),
        );
        assert!(matches!(
            query.match_type(),
            FastFieldStrMatchType::StartsWith(_)
        ));

        let query = FastFieldStrQuery::ends_with(
            Term::from_field_text(title_field, ""),
            "test".to_string(),
        );
        assert!(matches!(
            query.match_type(),
            FastFieldStrMatchType::EndsWith(_)
        ));

        let query = FastFieldStrQuery::regex(
            Term::from_field_text(title_field, ""),
            Regex::new("test").unwrap(),
        );
        assert!(matches!(
            query.match_type(),
            FastFieldStrMatchType::Regex(_)
        ));

        Ok(())
    }

    #[test]
    fn test_new_constructor() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", STRING | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(title_field => "hello world"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::new(
            Term::from_field_text(title_field, ""),
            FastFieldStrMatchType::Contains("hello".to_string()),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        Ok(())
    }

    #[test]
    fn test_file_extensions() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let filename_field = schema_builder.add_text_field("filename", STRING | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(filename_field => "document.pdf"))?;
        index_writer.add_document(doc!(filename_field => "report.PDF"))?;
        index_writer.add_document(doc!(filename_field => "image.png"))?;
        index_writer.add_document(doc!(filename_field => "script.exe"))?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldStrQuery::ends_with(
            Term::from_field_text(filename_field, ""),
            ".pdf".to_string(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let mut query = FastFieldStrQuery::ends_with(
            Term::from_field_text(filename_field, ""),
            ".pdf".to_string(),
        );
        query.set_case_insensitive(true);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query = FastFieldStrQuery::regex(
            Term::from_field_text(filename_field, ""),
            Regex::new("(?i)\\.pdf$").unwrap(),
        );
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        Ok(())
    }
}
