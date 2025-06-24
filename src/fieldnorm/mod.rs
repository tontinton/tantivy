//! The fieldnorm represents the length associated with
//! a given Field of a given document.
//!
//! This metric is important to compute the score of a
//! document: a document having a query word in one of its short fields
//! (e.g. title)  is likely to be more relevant than in one of its longer field
//! (e.g. body).
//!
//! It encodes `fieldnorm` on one byte with some precision loss,
//! using the exact same scheme as Lucene. Each value is placed on a log-scale
//! that takes values from `0` to `255`.
//!
//! A value on this scale is identified by a `fieldnorm_id`.
//! Apart from compression, this scale also makes it possible to
//! precompute computationally expensive functions of the fieldnorm
//! in a very short array.
//!
//! This trick is used by the Bm25 similarity.
mod code;
mod compression;
mod reader;
mod serializer;
mod writer;

use self::code::{fieldnorm_to_id, id_to_fieldnorm};
pub use self::reader::{FieldNormReader, FieldNormReaders};
pub use self::serializer::FieldNormsSerializer;
pub use self::writer::FieldNormsWriter;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use once_cell::sync::Lazy;

    use crate::directory::{CompositeFile, Directory, RamDirectory, WritePtr};
    use crate::fieldnorm::{FieldNormReader, FieldNormsSerializer, FieldNormsWriter};
    use crate::query::{EnableScoring, Query, TermQuery};
    use crate::schema::{
        Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, STORED, TEXT,
    };
    use crate::store::Compressor;
    use crate::{Index, Term, TERMINATED};

    pub const DYNAMIC_FIELD_NAME: &str = "_dynamic";

    pub static SCHEMA: Lazy<Schema> = Lazy::new(|| {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("field", STORED);
        schema_builder.add_text_field("txt_field", TEXT);
        schema_builder.add_text_field(
            "str_field",
            TextOptions::default().set_indexing_options(
                TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::Basic)
                    .set_fieldnorms(false),
            ),
        );
        schema_builder.add_json_field(DYNAMIC_FIELD_NAME, TEXT);
        schema_builder.add_json_field("json_not_indexed", STORED);
        schema_builder.build()
    });

    pub static FIELD: Lazy<Field> = Lazy::new(|| SCHEMA.get_field("field").unwrap());
    pub static TXT_FIELD: Lazy<Field> = Lazy::new(|| SCHEMA.get_field("txt_field").unwrap());
    pub static STR_FIELD: Lazy<Field> = Lazy::new(|| SCHEMA.get_field("str_field").unwrap());
    pub static JSON_FIELD: Lazy<Field> =
        Lazy::new(|| SCHEMA.get_field(DYNAMIC_FIELD_NAME).unwrap());
    pub static JSON_NOT_INDEXED_FIELD: Lazy<Field> =
        Lazy::new(|| SCHEMA.get_field("json_not_indexed").unwrap());

    #[test]
    #[should_panic(expected = "Cannot register a given fieldnorm twice")]
    pub fn test_should_panic_when_recording_fieldnorm_twice_for_same_doc() {
        let mut fieldnorm_writers = FieldNormsWriter::for_schema(&SCHEMA);
        fieldnorm_writers.record(0u32, *TXT_FIELD, 5);
        fieldnorm_writers.record(0u32, *TXT_FIELD, 3);
    }

    #[test]
    pub fn test_fieldnorm() -> crate::Result<()> {
        let path = Path::new("test");
        let directory: RamDirectory = RamDirectory::create();
        {
            let write: WritePtr = directory.open_write(Path::new("test"))?;
            let serializer = FieldNormsSerializer::from_write(write, Compressor::None)?;
            let mut fieldnorm_writers = FieldNormsWriter::for_schema(&SCHEMA);
            fieldnorm_writers.record(2u32, *TXT_FIELD, 5);
            fieldnorm_writers.record(3u32, *TXT_FIELD, 3);
            fieldnorm_writers.serialize(serializer)?;
        }
        let file = directory.open_read(path)?;
        {
            let fields_composite: CompositeFile<usize> = CompositeFile::open(&file)?;
            assert!(fields_composite.open_read(*FIELD).is_none());
            assert!(fields_composite.open_read(*STR_FIELD).is_none());
            let data = fields_composite.open_read(*TXT_FIELD).unwrap();
            let fieldnorm_reader = FieldNormReader::open(data)?;
            assert_eq!(fieldnorm_reader.fieldnorm(0u32), 0u32);
            assert_eq!(fieldnorm_reader.fieldnorm(1u32), 0u32);
            assert_eq!(fieldnorm_reader.fieldnorm(2u32), 5u32);
            assert_eq!(fieldnorm_reader.fieldnorm(3u32), 3u32);
        }
        Ok(())
    }

    #[test]
    pub fn test_fieldnorm_compression() -> crate::Result<()> {
        let directory: RamDirectory = RamDirectory::create();
        let path_with_compression = Path::new("test_compressed");
        let path_without_compression = Path::new("test_uncompressed");
        {
            let serializer_with_compression = FieldNormsSerializer::from_write(
                directory.open_write(path_with_compression)?,
                Compressor::Lz4,
            )?;
            let serializer_without_compression = FieldNormsSerializer::from_write(
                directory.open_write(path_without_compression)?,
                Compressor::None,
            )?;

            let mut fieldnorm_writers = FieldNormsWriter::for_schema(&SCHEMA);
            fieldnorm_writers.record(1u32, *TXT_FIELD, 5);
            fieldnorm_writers.record(2u32, *TXT_FIELD, 3);
            fieldnorm_writers.record_json(1u32, *JSON_FIELD, "title", 4); // Vega ran by ducks
            fieldnorm_writers.record_json(1u32, *JSON_FIELD, "body", 1); // Quack
            fieldnorm_writers.record_json(2u32, *JSON_FIELD, "question", 5); // Who let the ducks code?
            fieldnorm_writers.fill_up_to_max_doc(3u32);

            fieldnorm_writers.serialize(serializer_with_compression)?;
            let uncompressed_fieldnorms_size = directory.total_mem_usage();
            fieldnorm_writers.serialize(serializer_without_compression)?;
            let compressed_fieldnorms_size =
                directory.total_mem_usage() - uncompressed_fieldnorms_size;
            assert!(compressed_fieldnorms_size < uncompressed_fieldnorms_size);
        }

        let file = directory.open_read(path_with_compression)?;
        {
            let fields_composite: CompositeFile<String> = CompositeFile::open(&file)?;
            let json_title_fieldnorm_reader = FieldNormReader::open(
                fields_composite
                    .open_read_with_idx(*JSON_FIELD, "title".to_string())
                    .unwrap(),
            )?;
            let text_fieldnorm_reader =
                FieldNormReader::open(fields_composite.open_read(*TXT_FIELD).unwrap())?;

            assert_eq!(text_fieldnorm_reader.fieldnorm(0u32), 0u32);
            assert_eq!(text_fieldnorm_reader.fieldnorm(1u32), 5u32);
            assert_eq!(text_fieldnorm_reader.fieldnorm(2u32), 3u32);

            assert_eq!(json_title_fieldnorm_reader.fieldnorm(0u32), 0u32);
            assert_eq!(json_title_fieldnorm_reader.fieldnorm(1u32), 4u32);
            assert_eq!(json_title_fieldnorm_reader.fieldnorm(2u32), 0u32);
        }
        Ok(())
    }

    #[test]
    pub fn test_json_fieldnorm() -> crate::Result<()> {
        let path = Path::new("test");
        let directory: RamDirectory = RamDirectory::create();
        {
            let write: WritePtr = directory.open_write(Path::new("test"))?;
            let serializer = FieldNormsSerializer::from_write(write, Compressor::None)?;
            let mut fieldnorm_writers = FieldNormsWriter::for_schema(&SCHEMA);
            fieldnorm_writers.record_json(1u32, *JSON_FIELD, "title", 4); // Vega ran by ducks
            fieldnorm_writers.record_json(1u32, *JSON_FIELD, "body", 1); // Quack
            fieldnorm_writers.record_json(2u32, *JSON_FIELD, "question", 5); // Who let the ducks code?
            fieldnorm_writers.fill_up_to_max_doc(3u32);
            fieldnorm_writers.serialize(serializer)?;
        }
        let file = directory.open_read(path)?;
        {
            let fields_composite: CompositeFile<String> = CompositeFile::open(&file)?;
            assert!(fields_composite
                .open_read(*JSON_NOT_INDEXED_FIELD)
                .is_none());
            let title_fieldnorm_reader = FieldNormReader::open(
                fields_composite
                    .open_read_with_idx(*JSON_FIELD, "title".to_string())
                    .unwrap(),
            )?;
            let body_fieldnorm_reader = FieldNormReader::open(
                fields_composite
                    .open_read_with_idx(*JSON_FIELD, "body".to_string())
                    .unwrap(),
            )?;
            let question_fieldnorm_reader = FieldNormReader::open(
                fields_composite
                    .open_read_with_idx(*JSON_FIELD, "question".to_string())
                    .unwrap(),
            )?;

            assert_eq!(title_fieldnorm_reader.fieldnorm(0u32), 0u32);
            assert_eq!(body_fieldnorm_reader.fieldnorm(0u32), 0u32);
            assert_eq!(question_fieldnorm_reader.fieldnorm(0u32), 0u32);

            assert_eq!(title_fieldnorm_reader.fieldnorm(1u32), 4u32);
            assert_eq!(body_fieldnorm_reader.fieldnorm(1u32), 1u32);
            assert_eq!(question_fieldnorm_reader.fieldnorm(1u32), 0u32);

            assert_eq!(title_fieldnorm_reader.fieldnorm(2u32), 0u32);
            assert_eq!(body_fieldnorm_reader.fieldnorm(2u32), 0u32);
            assert_eq!(question_fieldnorm_reader.fieldnorm(2u32), 5u32);
        }
        Ok(())
    }

    #[test]
    fn test_fieldnorm_disabled() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_options = TextOptions::default()
            .set_indexing_options(TextFieldIndexing::default().set_fieldnorms(false));
        let text = schema_builder.add_text_field("text", text_options);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests()?;
        writer.add_document(doc!(text=>"hello"))?;
        writer.add_document(doc!(text=>"hello hello hello"))?;
        writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = TermQuery::new(
            Term::from_field_text(text, "hello"),
            IndexRecordOption::WithFreqs,
        );
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0f32)?;
        assert_eq!(scorer.doc(), 0);
        assert!((scorer.score() - 0.22920431).abs() < 0.001f32);
        assert_eq!(scorer.advance(), 1);
        assert_eq!(scorer.doc(), 1);
        assert!((scorer.score() - 0.22920431).abs() < 0.001f32);
        assert_eq!(scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_fieldnorm_enabled() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_options = TextOptions::default()
            .set_indexing_options(TextFieldIndexing::default().set_fieldnorms(true));
        let text = schema_builder.add_text_field("text", text_options);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests()?;
        writer.add_document(doc!(text=>"hello"))?;
        writer.add_document(doc!(text=>"hello hello hello"))?;
        writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = TermQuery::new(
            Term::from_field_text(text, "hello"),
            IndexRecordOption::WithFreqs,
        );
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0f32)?;
        assert_eq!(scorer.doc(), 0);
        assert!((scorer.score() - 0.22920431).abs() < 0.001f32);
        assert_eq!(scorer.advance(), 1);
        assert_eq!(scorer.doc(), 1);
        assert!((scorer.score() - 0.15136132).abs() < 0.001f32);
        assert_eq!(scorer.advance(), TERMINATED);
        Ok(())
    }
}
