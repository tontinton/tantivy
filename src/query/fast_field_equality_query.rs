//! Fast field equality query for all column types.
//!
//! This module provides an efficient query for checking equality on fast fields.
//! It supports all fast field types: strings, numbers (u64, i64, f64), booleans,
//! IP addresses, and dates.
//!
//! For strings with case-sensitive matching, it leverages dictionary encoding:
//! 1. Look up the target string in the dictionary to get its term ordinal (once)
//! 2. Scan the ordinal column for matching ordinals (using SIMD-optimized range scan)
//!
//! For strings with case-insensitive matching, it iterates all dictionary terms.
//!
//! For numeric types, booleans, IPs, and dates, it uses a range scan with equal bounds.

use std::fmt;
use std::net::Ipv6Addr;

use columnar::{Column, ColumnType, MonotonicallyMappableToU64, StrColumn};

use super::range_query::fast_field_range_doc_set::RangeDocSet;
use crate::query::fast_field_str_common::{collect_matching_ords, matching_ords_to_scorer};
use crate::query::{ConstScorer, EmptyScorer, EnableScoring, Explanation, Query, Scorer, Weight};
use crate::schema::Field;
use crate::{DateTime, DocId, DocSet, Score, SegmentReader, TantivyError, TERMINATED};

/// Value types supported by FastFieldEqualityQuery.
#[derive(Clone, Debug, PartialEq)]
pub enum FastFieldEqualityValue {
    /// String value.
    Str(String),
    /// Unsigned 64-bit integer value.
    U64(u64),
    /// Signed 64-bit integer value.
    I64(i64),
    /// 64-bit floating point value.
    F64(f64),
    /// Boolean value.
    Bool(bool),
    /// IPv6 address value (IPv4 addresses are mapped to IPv6).
    IpAddr(Ipv6Addr),
    /// Date/time value.
    Date(DateTime),
}

impl fmt::Display for FastFieldEqualityValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FastFieldEqualityValue::Str(v) => write!(f, "\"{}\"", v),
            FastFieldEqualityValue::U64(v) => write!(f, "{}", v),
            FastFieldEqualityValue::I64(v) => write!(f, "{}", v),
            FastFieldEqualityValue::F64(v) => write!(f, "{}", v),
            FastFieldEqualityValue::Bool(v) => write!(f, "{}", v),
            FastFieldEqualityValue::IpAddr(v) => write!(f, "{}", v),
            FastFieldEqualityValue::Date(v) => write!(f, "{:?}", v),
        }
    }
}

impl From<String> for FastFieldEqualityValue {
    fn from(v: String) -> Self {
        FastFieldEqualityValue::Str(v)
    }
}

impl From<&str> for FastFieldEqualityValue {
    fn from(v: &str) -> Self {
        FastFieldEqualityValue::Str(v.to_string())
    }
}

impl From<u64> for FastFieldEqualityValue {
    fn from(v: u64) -> Self {
        FastFieldEqualityValue::U64(v)
    }
}

impl From<i64> for FastFieldEqualityValue {
    fn from(v: i64) -> Self {
        FastFieldEqualityValue::I64(v)
    }
}

impl From<f64> for FastFieldEqualityValue {
    fn from(v: f64) -> Self {
        FastFieldEqualityValue::F64(v)
    }
}

impl From<bool> for FastFieldEqualityValue {
    fn from(v: bool) -> Self {
        FastFieldEqualityValue::Bool(v)
    }
}

impl From<Ipv6Addr> for FastFieldEqualityValue {
    fn from(v: Ipv6Addr) -> Self {
        FastFieldEqualityValue::IpAddr(v)
    }
}

impl From<std::net::IpAddr> for FastFieldEqualityValue {
    fn from(v: std::net::IpAddr) -> Self {
        FastFieldEqualityValue::IpAddr(v.into_ipv6_addr())
    }
}

impl From<DateTime> for FastFieldEqualityValue {
    fn from(v: DateTime) -> Self {
        FastFieldEqualityValue::Date(v)
    }
}

use crate::schema::IntoIpv6Addr;

/// A query that matches documents where a fast field equals a specific value.
///
/// This query is optimized for fast fields and supports all column types:
/// - Strings (with optional case-insensitive matching)
/// - Numbers (u64, i64, f64)
/// - Booleans
/// - IP addresses
/// - Dates
#[derive(Clone)]
pub struct FastFieldEqualityQuery {
    field: Field,
    value: FastFieldEqualityValue,
    case_insensitive: bool,
}

impl fmt::Debug for FastFieldEqualityQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FastFieldEqualityQuery")
            .field("field", &self.field)
            .field("value", &self.value)
            .field("case_insensitive", &self.case_insensitive)
            .finish()
    }
}

impl FastFieldEqualityQuery {
    /// Creates a new `FastFieldEqualityQuery`.
    ///
    /// # Arguments
    ///
    /// * `field` - The field to search on. Must have the FAST flag.
    /// * `value` - The value to match.
    pub fn new<V: Into<FastFieldEqualityValue>>(field: Field, value: V) -> Self {
        Self {
            field,
            value: value.into(),
            case_insensitive: false,
        }
    }

    /// Creates a new case-insensitive `FastFieldEqualityQuery` for string values.
    ///
    /// # Arguments
    ///
    /// * `field` - The field to search on. Must be a string field with the FAST flag.
    /// * `value` - The string value to match (will be matched case-insensitively).
    ///
    /// # Panics
    ///
    /// Panics if the value is not a string type.
    pub fn new_str_case_insensitive(field: Field, value: String) -> Self {
        Self {
            field,
            value: FastFieldEqualityValue::Str(value),
            case_insensitive: true,
        }
    }

    /// Sets whether the search should be case-insensitive (only applies to string values).
    pub fn set_case_insensitive(&mut self, case_insensitive: bool) -> &mut Self {
        self.case_insensitive = case_insensitive;
        self
    }

    /// Returns whether this query is case-insensitive.
    pub fn is_case_insensitive(&self) -> bool {
        self.case_insensitive
    }

    /// Returns the field being searched.
    pub fn field(&self) -> Field {
        self.field
    }

    /// Returns the value being matched.
    pub fn value(&self) -> &FastFieldEqualityValue {
        &self.value
    }
}

impl Query for FastFieldEqualityQuery {
    fn weight(&self, _enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(FastFieldEqualityWeight {
            field: self.field,
            value: self.value.clone(),
            case_insensitive: self.case_insensitive,
        }))
    }
}

/// Weight for the fast field equality query.
#[derive(Clone)]
pub struct FastFieldEqualityWeight {
    field: Field,
    value: FastFieldEqualityValue,
    case_insensitive: bool,
}

impl fmt::Debug for FastFieldEqualityWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FastFieldEqualityWeight")
            .field("field", &self.field)
            .field("value", &self.value)
            .field("case_insensitive", &self.case_insensitive)
            .finish()
    }
}

impl FastFieldEqualityWeight {
    fn get_field_name(&self, reader: &SegmentReader) -> crate::Result<String> {
        let schema = reader.schema();
        let field_entry = schema.get_field_entry(self.field);
        let field_name = field_entry.name();

        let field_type = field_entry.field_type();
        if !field_type.is_fast() {
            return Err(TantivyError::InvalidArgument(format!(
                "Field '{field_name}' is not a fast field"
            )));
        }

        Ok(field_name.to_string())
    }

    fn scorer_str(
        &self,
        reader: &SegmentReader,
        boost: Score,
        value: &str,
    ) -> crate::Result<Box<dyn Scorer>> {
        let field_name = self.get_field_name(reader)?;

        let str_column: Option<StrColumn> = reader.fast_fields().str(&field_name)?;
        let Some(str_column) = str_column else {
            return Ok(Box::new(EmptyScorer));
        };

        if self.case_insensitive {
            let value_lower = value.to_lowercase();
            let matching_ords = collect_matching_ords(&str_column, &field_name, |term_text| {
                term_text.to_lowercase() == value_lower
            })?;

            Ok(matching_ords_to_scorer(
                &str_column,
                &matching_ords,
                reader.max_doc(),
                boost,
            ))
        } else {
            let dictionary = str_column.dictionary();
            let term_ord = dictionary.term_ord(value.as_bytes())?;

            let Some(term_ord) = term_ord else {
                return Ok(Box::new(EmptyScorer));
            };

            let fast_field_reader = reader.fast_fields();
            let column_opt =
                fast_field_reader.u64_lenient_for_type(Some(&[ColumnType::Str]), &field_name)?;

            let Some((column, _col_type)) = column_opt else {
                return Ok(Box::new(EmptyScorer));
            };

            let docset = RangeDocSet::new(term_ord..=term_ord, column);
            Ok(Box::new(ConstScorer::new(docset, boost)))
        }
    }

    fn scorer_u64(
        &self,
        reader: &SegmentReader,
        boost: Score,
        value: u64,
    ) -> crate::Result<Box<dyn Scorer>> {
        let field_name = self.get_field_name(reader)?;
        let fast_field_reader = reader.fast_fields();

        let column_opt = fast_field_reader.u64_lenient_for_type(
            Some(&[ColumnType::U64, ColumnType::I64, ColumnType::F64]),
            &field_name,
        )?;

        let Some((column, col_type)) = column_opt else {
            return Ok(Box::new(EmptyScorer));
        };

        let mapped_value = match col_type {
            ColumnType::U64 => value,
            ColumnType::I64 => {
                if value > i64::MAX as u64 {
                    return Ok(Box::new(EmptyScorer));
                }
                (value as i64).to_u64()
            }
            ColumnType::F64 => (value as f64).to_u64(),
            _ => return Ok(Box::new(EmptyScorer)),
        };

        let docset = RangeDocSet::new(mapped_value..=mapped_value, column);
        Ok(Box::new(ConstScorer::new(docset, boost)))
    }

    fn scorer_i64(
        &self,
        reader: &SegmentReader,
        boost: Score,
        value: i64,
    ) -> crate::Result<Box<dyn Scorer>> {
        let field_name = self.get_field_name(reader)?;
        let fast_field_reader = reader.fast_fields();

        let column_opt = fast_field_reader.u64_lenient_for_type(
            Some(&[ColumnType::U64, ColumnType::I64, ColumnType::F64]),
            &field_name,
        )?;

        let Some((column, col_type)) = column_opt else {
            return Ok(Box::new(EmptyScorer));
        };

        let mapped_value = match col_type {
            ColumnType::I64 => value.to_u64(),
            ColumnType::U64 => {
                if value < 0 {
                    return Ok(Box::new(EmptyScorer));
                }
                value as u64
            }
            ColumnType::F64 => (value as f64).to_u64(),
            _ => return Ok(Box::new(EmptyScorer)),
        };

        let docset = RangeDocSet::new(mapped_value..=mapped_value, column);
        Ok(Box::new(ConstScorer::new(docset, boost)))
    }

    fn scorer_f64(
        &self,
        reader: &SegmentReader,
        boost: Score,
        value: f64,
    ) -> crate::Result<Box<dyn Scorer>> {
        let field_name = self.get_field_name(reader)?;
        let fast_field_reader = reader.fast_fields();

        let column_opt = fast_field_reader.u64_lenient_for_type(
            Some(&[ColumnType::U64, ColumnType::I64, ColumnType::F64]),
            &field_name,
        )?;

        let Some((column, col_type)) = column_opt else {
            return Ok(Box::new(EmptyScorer));
        };

        let mapped_value = match col_type {
            ColumnType::F64 => value.to_u64(),
            ColumnType::I64 => {
                if value.fract() != 0.0 || value < i64::MIN as f64 || value > i64::MAX as f64 {
                    return Ok(Box::new(EmptyScorer));
                }
                (value as i64).to_u64()
            }
            ColumnType::U64 => {
                if value.fract() != 0.0 || value < 0.0 || value > u64::MAX as f64 {
                    return Ok(Box::new(EmptyScorer));
                }
                value as u64
            }
            _ => return Ok(Box::new(EmptyScorer)),
        };

        let docset = RangeDocSet::new(mapped_value..=mapped_value, column);
        Ok(Box::new(ConstScorer::new(docset, boost)))
    }

    fn scorer_bool(
        &self,
        reader: &SegmentReader,
        boost: Score,
        value: bool,
    ) -> crate::Result<Box<dyn Scorer>> {
        let field_name = self.get_field_name(reader)?;

        let column: Option<Column<bool>> = reader.fast_fields().column_opt(&field_name)?;
        let Some(column) = column else {
            return Ok(Box::new(EmptyScorer));
        };

        let docset = RangeDocSet::new(value..=value, column);
        Ok(Box::new(ConstScorer::new(docset, boost)))
    }

    fn scorer_ip_addr(
        &self,
        reader: &SegmentReader,
        boost: Score,
        value: Ipv6Addr,
    ) -> crate::Result<Box<dyn Scorer>> {
        let field_name = self.get_field_name(reader)?;

        let column: Option<Column<Ipv6Addr>> = reader.fast_fields().column_opt(&field_name)?;
        let Some(column) = column else {
            return Ok(Box::new(EmptyScorer));
        };

        let docset = RangeDocSet::new(value..=value, column);
        Ok(Box::new(ConstScorer::new(docset, boost)))
    }

    fn scorer_date(
        &self,
        reader: &SegmentReader,
        boost: Score,
        value: DateTime,
    ) -> crate::Result<Box<dyn Scorer>> {
        let field_name = self.get_field_name(reader)?;

        let column: Option<Column<DateTime>> = reader.fast_fields().column_opt(&field_name)?;
        let Some(column) = column else {
            return Ok(Box::new(EmptyScorer));
        };

        let docset = RangeDocSet::new(value..=value, column);
        Ok(Box::new(ConstScorer::new(docset, boost)))
    }
}

impl Weight for FastFieldEqualityWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        match &self.value {
            FastFieldEqualityValue::Str(v) => self.scorer_str(reader, boost, v),
            FastFieldEqualityValue::U64(v) => self.scorer_u64(reader, boost, *v),
            FastFieldEqualityValue::I64(v) => self.scorer_i64(reader, boost, *v),
            FastFieldEqualityValue::F64(v) => self.scorer_f64(reader, boost, *v),
            FastFieldEqualityValue::Bool(v) => self.scorer_bool(reader, boost, *v),
            FastFieldEqualityValue::IpAddr(v) => self.scorer_ip_addr(reader, boost, *v),
            FastFieldEqualityValue::Date(v) => self.scorer_date(reader, boost, *v),
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(TantivyError::InvalidArgument(format!(
                "Document #({doc}) does not match"
            )));
        }
        Ok(Explanation::new("FastFieldEquality", scorer.score()))
    }

    fn count(&self, reader: &SegmentReader) -> crate::Result<u32> {
        let mut scorer = self.scorer(reader, 1.0)?;
        let mut count = 0u32;
        while scorer.doc() != TERMINATED {
            count += 1;
            scorer.advance();
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use crate::collector::Count;
    use crate::query::FastFieldEqualityQuery;
    use crate::schema::{DateOptions, IpAddrOptions, NumericOptions, Schema, FAST, STORED, STRING};
    use crate::{DateTime, Index, IndexWriter};

    #[test]
    fn test_fast_field_equality_str() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let event_field = schema_builder.add_text_field("event", STRING | FAST);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;

        writer.add_document(doc!(event_field => "click"))?;
        writer.add_document(doc!(event_field => "view"))?;
        writer.add_document(doc!(event_field => "click"))?;
        writer.add_document(doc!(event_field => "purchase"))?;
        writer.add_document(doc!(event_field => "click"))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldEqualityQuery::new(event_field, "click");
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);

        let query = FastFieldEqualityQuery::new(event_field, "view");
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let query = FastFieldEqualityQuery::new(event_field, "nonexistent");
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_fast_field_equality_str_case_insensitive() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let event_field = schema_builder.add_text_field("event", STRING | FAST);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;

        writer.add_document(doc!(event_field => "Click"))?;
        writer.add_document(doc!(event_field => "CLICK"))?;
        writer.add_document(doc!(event_field => "click"))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldEqualityQuery::new(event_field, "click");
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let query =
            FastFieldEqualityQuery::new_str_case_insensitive(event_field, "click".to_string());
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);

        Ok(())
    }

    #[test]
    fn test_fast_field_equality_u64() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let count_field =
            schema_builder.add_u64_field("count", NumericOptions::default().set_fast());
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;

        writer.add_document(doc!(count_field => 10u64))?;
        writer.add_document(doc!(count_field => 20u64))?;
        writer.add_document(doc!(count_field => 10u64))?;
        writer.add_document(doc!(count_field => 30u64))?;
        writer.add_document(doc!(count_field => 10u64))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldEqualityQuery::new(count_field, 10u64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);

        let query = FastFieldEqualityQuery::new(count_field, 20u64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let query = FastFieldEqualityQuery::new(count_field, 999u64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_fast_field_equality_i64() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let score_field =
            schema_builder.add_i64_field("score", NumericOptions::default().set_fast());
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;

        writer.add_document(doc!(score_field => -10i64))?;
        writer.add_document(doc!(score_field => 20i64))?;
        writer.add_document(doc!(score_field => -10i64))?;
        writer.add_document(doc!(score_field => 0i64))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldEqualityQuery::new(score_field, -10i64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query = FastFieldEqualityQuery::new(score_field, 0i64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let query = FastFieldEqualityQuery::new(score_field, 999i64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_fast_field_equality_f64() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let price_field =
            schema_builder.add_f64_field("price", NumericOptions::default().set_fast());
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;

        writer.add_document(doc!(price_field => 9.99f64))?;
        writer.add_document(doc!(price_field => 19.99f64))?;
        writer.add_document(doc!(price_field => 9.99f64))?;
        writer.add_document(doc!(price_field => 0.0f64))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldEqualityQuery::new(price_field, 9.99f64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query = FastFieldEqualityQuery::new(price_field, 0.0f64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let query = FastFieldEqualityQuery::new(price_field, 999.99f64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_fast_field_equality_bool() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let active_field =
            schema_builder.add_bool_field("active", NumericOptions::default().set_fast());
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;

        writer.add_document(doc!(active_field => true))?;
        writer.add_document(doc!(active_field => false))?;
        writer.add_document(doc!(active_field => true))?;
        writer.add_document(doc!(active_field => true))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldEqualityQuery::new(active_field, true);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);

        let query = FastFieldEqualityQuery::new(active_field, false);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        Ok(())
    }

    #[test]
    fn test_fast_field_equality_ip_addr() -> crate::Result<()> {
        use std::net::Ipv6Addr;

        let mut schema_builder = Schema::builder();
        let ip_field = schema_builder.add_ip_addr_field("ip", IpAddrOptions::default().set_fast());
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;

        let ip1 = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc0a8, 0x0101);
        let ip2 = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc0a8, 0x0102);

        writer.add_document(doc!(ip_field => ip1))?;
        writer.add_document(doc!(ip_field => ip2))?;
        writer.add_document(doc!(ip_field => ip1))?;
        writer.add_document(doc!(ip_field => ip1))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldEqualityQuery::new(ip_field, ip1);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);

        let query = FastFieldEqualityQuery::new(ip_field, ip2);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let ip3 = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x0a00, 0x0001);
        let query = FastFieldEqualityQuery::new(ip_field, ip3);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_fast_field_equality_date() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let date_field =
            schema_builder.add_date_field("timestamp", DateOptions::default().set_fast());
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;

        let date1 = DateTime::from_timestamp_micros(1_000_000);
        let date2 = DateTime::from_timestamp_micros(2_000_000);

        writer.add_document(doc!(date_field => date1))?;
        writer.add_document(doc!(date_field => date2))?;
        writer.add_document(doc!(date_field => date1))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldEqualityQuery::new(date_field, date1);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        let query = FastFieldEqualityQuery::new(date_field, date2);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 1);

        let date3 = DateTime::from_timestamp_micros(3_000_000);
        let query = FastFieldEqualityQuery::new(date_field, date3);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_fast_field_equality_empty_index() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let count_field =
            schema_builder.add_u64_field("count", NumericOptions::default().set_fast());
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let writer: IndexWriter = index.writer_for_tests()?;
        drop(writer);

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldEqualityQuery::new(count_field, 10u64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_fast_field_equality_multiple_segments() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let count_field =
            schema_builder.add_u64_field("count", NumericOptions::default().set_fast());
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;

        writer.add_document(doc!(count_field => 10u64))?;
        writer.add_document(doc!(count_field => 20u64))?;
        writer.commit()?;

        writer.add_document(doc!(count_field => 10u64))?;
        writer.add_document(doc!(count_field => 30u64))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        assert_eq!(searcher.segment_readers().len(), 2);

        let query = FastFieldEqualityQuery::new(count_field, 10u64);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 2);

        Ok(())
    }

    #[test]
    fn test_fast_field_equality_non_fast_field_error() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let count_field = schema_builder.add_u64_field("count", STORED);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;
        writer.add_document(doc!(count_field => 10u64))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query = FastFieldEqualityQuery::new(count_field, 10u64);
        let result = searcher.search(&query, &Count);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("is not a fast field"));

        Ok(())
    }
}
