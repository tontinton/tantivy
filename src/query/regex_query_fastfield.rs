use std::io;

use columnar::{ColumnIndex, StrColumn};
use common::BitSet;
use regex::Regex;

use crate::query::explanation::does_not_match;
use crate::query::{
    BitSetDocSet, ConstScorer, EmptyScorer, EnableScoring, Explanation, Query, Scorer, Weight,
};
use crate::schema::{Term, Type};
use crate::{DocId, Score, SegmentReader, TantivyError};

#[derive(Debug, Clone)]
/// `FastFieldRegexQuery` is the same as [RegexQuery] but only uses the fast field
pub struct FastFieldRegexQuery {
    regex: Regex,
    term: Term, // term value is empty and ignored as it's the regex field
}

impl FastFieldRegexQuery {
    /// Create new `FastFieldRegexQuery`
    pub fn new(regex: Regex, term: Term) -> FastFieldRegexQuery {
        Self { regex, term }
    }
}

impl Query for FastFieldRegexQuery {
    fn weight(&self, _enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(FastFieldRegexWeight::new(
            self.regex.clone(),
            self.term.clone(),
        )))
    }
}

/// `FastFieldRegexWeight` uses the fast field to execute regex queries.
#[derive(Debug, Clone)]
pub struct FastFieldRegexWeight {
    regex: Regex,
    term: Term,
}

impl FastFieldRegexWeight {
    /// Create a new FastFieldRegexWeight
    pub fn new(regex: Regex, term: Term) -> Self {
        Self { regex, term }
    }
}

impl Weight for FastFieldRegexWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let schema = reader.schema();
        let field_type = schema.get_field_entry(self.term.field()).field_type();
        match field_type.value_type() {
            Type::Str => {}
            Type::Json => {
                self.term
                    .value()
                    .as_json_value_bytes()
                    .expect("expected json type in term");
            }
            _ => {
                return Err(TantivyError::InvalidArgument(
                    "regex query supports only on string or json fields".to_string(),
                ));
            }
        }

        let field_name = self.term.get_full_path(schema);
        let Some(str_dict_column): Option<StrColumn> = reader.fast_fields().str(&field_name)?
        else {
            return Ok(Box::new(EmptyScorer));
        };

        let mut matching_ords = BitSet::with_max_value(str_dict_column.num_terms() as u32);
        let result = str_dict_column.visit_all_ord_terms(|ord, term| {
            let term_text = std::str::from_utf8(term);
            if term_text.is_err() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("one of the fast field values for {field_name} is not valid UTF-8"),
                ));
            };
            let term_text = term_text.unwrap();
            if self.regex.is_match(term_text) {
                matching_ords.insert(ord as u32);
            }
            Ok(())
        });

        if result.is_err() {
            return Err(TantivyError::InternalError(format!(
                "failed to visit all ord terms: {:?}",
                result.unwrap_err()
            )));
        }

        if matching_ords.len() == 0 {
            return Ok(Box::new(EmptyScorer));
        }

        let ords_column = str_dict_column.ords();
        let mut matched_docs = BitSet::with_max_value(str_dict_column.num_rows());

        if let ColumnIndex::Optional(optional_index) = &ords_column.index {
            // Optimized approach in case the field doesn't exist in all documents (optional) and no
            // need to iterate over all documents
            const BATCH_SIZE: usize = 256;
            let num_non_nulls = optional_index.num_non_nulls();
            let mut values_buffer = vec![0u64; BATCH_SIZE];
            let mut matching_ranks = Vec::with_capacity(BATCH_SIZE);

            for batch_start in (0..num_non_nulls).step_by(BATCH_SIZE) {
                let batch_end = (batch_start + BATCH_SIZE as u32).min(num_non_nulls);
                let batch_len = (batch_end - batch_start) as usize;

                ords_column
                    .values
                    .get_range(batch_start as u64, &mut values_buffer[..batch_len]);

                matching_ranks.clear();
                for (i, &ord) in values_buffer[..batch_len].iter().enumerate() {
                    if matching_ords.contains(ord as u32) {
                        matching_ranks.push(batch_start + i as u32);
                    }
                }

                if !matching_ranks.is_empty() {
                    optional_index.select_batch(&mut matching_ranks);
                    for &doc_id in &matching_ranks {
                        matched_docs.insert(doc_id);
                    }
                }
            }
        } else {
            // In cases the field exists in all documents (full), or type of array (multivalued), we
            // iterate over all documents
            for doc_id in 0..reader.max_doc() {
                if ords_column
                    .values_for_doc(doc_id)
                    .any(|ord| matching_ords.contains(ord as u32))
                {
                    matched_docs.insert(doc_id);
                }
            }
        }

        if matched_docs.len() == 0 {
            return Ok(Box::new(EmptyScorer));
        }

        let docset = BitSetDocSet::from(matched_docs);
        Ok(Box::new(ConstScorer::new(docset, boost)))
    }

    fn explain(&self, _reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        Err(does_not_match(doc))
    }
}

#[cfg(test)]
mod tests {
    use regex::Regex;
    use serde_json::json;

    use crate::collector::TopDocs;
    use crate::query::regex_query_fastfield::FastFieldRegexQuery;
    use crate::schema::{Schema, FAST, TEXT};
    use crate::{Index, Term};

    #[test]
    fn test_text_field_ff_regex_query() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title_field = schema_builder.add_text_field("title", TEXT | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut index_writer = index.writer_for_tests()?;
        let docs = vec![
            "cmd.exe",
            "powershell.exe",
            "powershel.exe",
            "c:\\users\\public\\",
            "c:\\users\\public\\documents\\",
            "c:\\users\\myuser\\public\\",
            "c:\\temp\\",
            "c:\\users\\temp\\doc.txt",
            "c:\\programdat\\temp\\doc.txt",
            "192.168.1.1",
            "169.254.1.1",
            "169.254.1.100",
            "169-254-1-100",
            "10.254.1.100",
            "11.254.1.100",
            "169.254.1.3100",
            "SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB2ZXJ5IGxvbmcgQmFzZTY0IGVuY29kZWQgc3RyaW5nIHRoYXQgY29udGFpbnMgbW9yZSB0aGFuIDIwMC==",
            "SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB2ZXJ5IGxvbmcgQmFzZTY0IGVuY29kZWQgc3RyaW5nIHRoYXQgY29udGFpbnMgbW9yZSB0aGFuIDIwMC",
            "powershell.exe -enc JABjAGwAaQBlAG4AdAAgAD0AIABOAGUAdwAtAE8AYgBqAGUAYwB0ACAAUwB5AHMAdABlAG0ALgBOAGUAdAAuAFMAbwBjAGsAZQB0AHMALgBUAEMAUABDAGwAaAsADAALAAgACQAaQApADsA",
            "powershell.exe -enccommand JABAGEAdABhACAAPQAA==",
            "powershell.exe -encodedcommand JABAGEAdABhACAA",
        ];

        for value in docs {
            index_writer.add_document(doc!(title_field => value))?;
        }
        index_writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let test_query = |regex, num_hits| {
            let query = FastFieldRegexQuery::new(
                Regex::new(regex).unwrap(),
                Term::from_field_text(title_field, ""),
            );
            let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
            assert_eq!(top_docs.len(), num_hits);
        };

        test_query("^(powershell|cmd)\\.exe$", 2);
        test_query("^(power shell|cmd)\\.exe$", 1);
        test_query("(?i)powershell.*-enc(oded)?command", 2);
        test_query(
            "(?i)c:\\\\users\\\\public\\\\|c:\\\\programdata\\\\|\\\\temp\\\\",
            5,
        );
        test_query("(?i)c:\\\\programdata\\\\", 0);
        test_query("[A-Za-z0-9+/]{50,}={0,2}", 3);
        test_query(
            r"^(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|169\.254\.\d{1,3}\.\d{1,3})$",
            4,
        );

        Ok(())
    }

    #[test]
    fn test_json_sub_field_ff_regex_query() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut index_writer = index.writer_for_tests()?;

        let docs = vec![
            json!({"user": {"command": "powershell.exe"}}),
            json!({"user": {"command": "cmd.exe"}}),
            json!({"user": {"command": "powershell.exe -enccommand"}}),
            json!({"user": {"path": "c:\\users\\public\\"}}),
            json!({"user": {"path": "c:\\ProgramData\\logs\\"}}),
            json!({"user": {"path": "c:\\ProgramData\\"}}),
        ];

        for value in docs {
            index_writer.add_document(doc!(json_field => value))?;
        }

        index_writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let mut command_term = Term::from_field_json_path(json_field, "user.command", false);
        command_term.append_type_and_str("");

        let mut path_term = Term::from_field_json_path(json_field, "user.path", false);
        path_term.append_type_and_str("");

        let run = |regex: &str, term: Term| -> usize {
            let query = FastFieldRegexQuery::new(Regex::new(regex).unwrap(), term);
            searcher
                .search(&query, &TopDocs::with_limit(10))
                .unwrap()
                .len()
        };

        assert_eq!(run("(?i)powershell", command_term.clone()), 2);
        assert_eq!(run("^cmd\\.exe$", command_term.clone()), 1);
        assert_eq!(run("(?i)c:\\\\programdata\\\\", path_term.clone()), 2);

        Ok(())
    }

    #[test]
    fn test_array_field_ff_regex_query() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let tags_field = schema_builder.add_text_field("tags", TEXT | FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut index_writer = index.writer_for_tests()?;

        index_writer.add_document(doc!(tags_field => "powershell", tags_field => "warn"))?;
        index_writer.add_document(doc!(tags_field => "debug", tags_field => "info"))?;
        index_writer.add_document(doc!(tags_field => "notice"))?;
        index_writer.add_document(doc!(tags_field => "warning", tags_field => "error"))?;

        index_writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let run = |regex: &str| -> usize {
            let query = FastFieldRegexQuery::new(
                Regex::new(regex).unwrap(),
                Term::from_field_text(tags_field, ""),
            );
            searcher
                .search(&query, &TopDocs::with_limit(10))
                .unwrap()
                .len()
        };

        assert_eq!(run("^warn$"), 1);
        assert_eq!(run("(?i)debug|error"), 2);
        assert_eq!(run("^(notice|info)$"), 2);
        assert_eq!(run("^powershell$"), 1);
        assert_eq!(run("nomatch"), 0);

        Ok(())
    }

    #[test]
    fn test_numeric_json_sub_field_ff_regex_query() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut index_writer = index.writer_for_tests()?;

        let docs = vec![json!({"user": {"num": 123}}), json!({"user": {"num": 456}})];

        for value in docs {
            index_writer.add_document(doc!(json_field => value))?;
        }

        index_writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let mut num_term = Term::from_field_json_path(json_field, "user.num", false);
        num_term.append_type_and_str("");
        let run = |regex: &str| -> usize {
            let query = FastFieldRegexQuery::new(Regex::new(regex).unwrap(), num_term);
            searcher
                .search(&query, &TopDocs::with_limit(10))
                .unwrap()
                .len()
        };

        assert_eq!(run("(?i)123"), 0);

        Ok(())
    }
}
