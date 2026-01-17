use columnar::{ColumnIndex, StrColumn};
use common::BitSet;

use crate::query::{BitSetDocSet, ConstScorer, EmptyScorer, Scorer};
use crate::schema::{Term, Type};
use crate::{Score, SegmentReader, TantivyError};

/// Validates that the term's field is a string or JSON type and returns the str column.
///
/// Returns `Ok(None)` if the column doesn't exist (empty scorer case).
/// Returns `Err` if the field type is invalid.
pub(crate) fn get_str_column_for_term(
    reader: &SegmentReader,
    term: &Term,
    query_name: &str,
) -> crate::Result<Option<(StrColumn, String)>> {
    let schema = reader.schema();
    let field_type = schema.get_field_entry(term.field()).field_type();
    match field_type.value_type() {
        Type::Str => {}
        Type::Json => {
            term.value()
                .as_json_value_bytes()
                .expect("expected json type in term");
        }
        _ => {
            return Err(TantivyError::InvalidArgument(format!(
                "{query_name} supports only string or json fields"
            )));
        }
    }

    let field_name = term.get_full_path(schema);
    let str_column: Option<StrColumn> = reader.fast_fields().str(&field_name)?;

    Ok(str_column.map(|col| (col, field_name)))
}

/// Maps matching term ordinals to document IDs and returns a scorer.
///
/// This function handles both optional fields (using optimized batch processing)
/// and full/multivalued fields (using linear iteration).
pub(crate) fn matching_ords_to_scorer(
    str_column: &StrColumn,
    matching_ords: &BitSet,
    max_doc: u32,
    boost: Score,
) -> Box<dyn Scorer> {
    if matching_ords.len() == 0 {
        return Box::new(EmptyScorer);
    }

    let ords_column = str_column.ords();
    let mut matched_docs = BitSet::with_max_value(str_column.num_rows());

    if let ColumnIndex::Optional(optional_index) = &ords_column.index {
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
        for doc_id in 0..max_doc {
            if ords_column
                .values_for_doc(doc_id)
                .any(|ord| matching_ords.contains(ord as u32))
            {
                matched_docs.insert(doc_id);
            }
        }
    }

    if matched_docs.len() == 0 {
        return Box::new(EmptyScorer);
    }

    let docset = BitSetDocSet::from(matched_docs);
    Box::new(ConstScorer::new(docset, boost))
}

/// Visits all terms in the dictionary and collects ordinals that match the predicate.
///
/// Returns `Err` if any term is not valid UTF-8 or if visiting fails.
pub(crate) fn collect_matching_ords<F>(
    str_column: &StrColumn,
    field_name: &str,
    mut matches: F,
) -> crate::Result<BitSet>
where
    F: FnMut(&str) -> bool,
{
    let mut matching_ords = BitSet::with_max_value(str_column.num_terms() as u32);

    let result = str_column.visit_all_ord_terms(|ord, term| {
        let term_text = std::str::from_utf8(term).map_err(|_| {
            std::io::Error::other(format!(
                "one of the fast field values for {field_name} is not valid UTF-8"
            ))
        })?;
        if matches(term_text) {
            matching_ords.insert(ord as u32);
        }
        Ok(())
    });

    if let Err(e) = result {
        return Err(TantivyError::InternalError(format!(
            "failed to visit all ord terms: {e:?}"
        )));
    }

    Ok(matching_ords)
}
