use common::rate_limited_warn;

use crate::query::explanation::does_not_match;
use crate::query::{EnableScoring, ExistsQuery, Explanation, Query, TermQuery, Weight};
use crate::schema::{Field, IndexRecordOption, Type};
use crate::{TantivyError, Term};

/// On json intermediates (i.e. not leafs), lookup with field_presence (TermQuery).
/// On json leafs, fallback to lookup with fast-field (ExistsQuery).
/// Therefore, the field must be a fast field.
#[derive(Clone, Debug)]
pub struct JsonExistsQuery {
    key: String,
    term_query: TermQuery,
}

impl JsonExistsQuery {
    /// Creates a new `JsonExistsQuery` from the given field.
    ///
    /// This constructor never fails, but executing the search with this query will
    /// return an error if the specified field doesn't exist or is not a fast
    /// field.
    pub fn new(key: String, value_hash: u64, field_presence: Field) -> Self {
        let intermediate_term = Term::from_field_u64(field_presence, value_hash);
        let term_query = TermQuery::new(intermediate_term, IndexRecordOption::Basic);

        Self { key, term_query }
    }
}

impl Query for JsonExistsQuery {
    fn weight(&self, enable_scoring: EnableScoring) -> crate::Result<Box<dyn Weight>> {
        let term_weight = self.term_query.weight(enable_scoring)?;
        let fallback_exists_weight =
            ExistsQuery::new(self.key.clone(), true).weight(enable_scoring)?;

        let schema = enable_scoring.schema();
        let Some((field, _path)) = schema.find_field(&self.key) else {
            return Err(TantivyError::FieldNotFound(self.key.clone()));
        };
        let field_type = schema.get_field_entry(field).field_type();
        return Ok(Box::new(JsonExistsWeight {
            term_weight,
            fallback_exists_weight,
            full_path: self.key.clone(),
            field_type: field_type.value_type(),
        }));
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        self.term_query.query_terms(visitor)
    }
}

pub struct JsonExistsWeight {
    term_weight: Box<dyn Weight>,
    fallback_exists_weight: Box<dyn Weight>,
    full_path: String,
    field_type: Type,
}

impl Weight for JsonExistsWeight {
    fn scorer(
        &self,
        reader: &crate::SegmentReader,
        boost: crate::Score,
    ) -> crate::Result<Box<dyn super::Scorer>> {
        let fast_field_reader = reader.fast_fields();
        if self.field_type == Type::Json {
            // intermediate i.e. not leaf.
            let is_intermediate = fast_field_reader.any_dynamic_subpath(&self.full_path)?;
            if is_intermediate {
                // term query on _field_presence_json.
                rate_limited_warn!(
                    limit_per_sec = 1,
                    "JsonExistsWeight: use term weight; search path: {}",
                    self.full_path
                );
                return self.term_weight.scorer(reader, boost);
            }
        }
        rate_limited_warn!(
            limit_per_sec = 1,
            "JsonExistsWeight: fallback to ExistsQuery; search path: {}",
            self.full_path
        );
        self.fallback_exists_weight.scorer(reader, boost)
    }

    fn explain(
        &self,
        reader: &crate::SegmentReader,
        doc: crate::DocId,
    ) -> crate::Result<super::Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        Ok(Explanation::new("JsonExistsQuery", 1.0))
    }
}
