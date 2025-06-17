use std::cmp::Ordering;
use std::{io, iter};

use hashbrown::HashMap;

use super::{fieldnorm_to_id, FieldNormsSerializer};
use crate::schema::{Field, Schema};
use crate::DocId;

/// The `FieldNormsWriter` is in charge of tracking the fieldnorm byte
/// of each document for each field with field norms.
///
/// `FieldNormsWriter` stores a `Vec<u8>` for each tracked field, using a
/// byte per document per field.
pub struct FieldNormsWriter {
    fieldnorms_buffers: Vec<Option<Vec<u8>>>,
    json_fieldnorms_buffers: HashMap<Field, HashMap<String, Vec<u8>>>,
}

impl FieldNormsWriter {
    fn fieldnorm_fields(schema: &Schema) -> impl Iterator<Item = (Field, bool)> + '_ {
        schema
            .fields()
            .filter(|(_, field_entry)| field_entry.is_indexed() && field_entry.has_fieldnorms())
            .map(|(field, field_entry)| (field, field_entry.field_type().is_json()))
    }

    /// Returns static & json fields that should have field norms computed according to the given
    /// schema.
    pub(crate) fn fields_with_fieldnorm(schema: &Schema) -> (Vec<Field>, Vec<Field>) {
        let mut static_fields = Vec::new();
        let mut json_fields = Vec::new();

        for (field, is_json) in Self::fieldnorm_fields(schema) {
            if is_json {
                json_fields.push(field);
            } else {
                static_fields.push(field);
            }
        }

        (static_fields, json_fields)
    }

    /// Initialize with state for tracking the field norm fields
    /// specified in the schema.
    pub fn for_schema(schema: &Schema) -> FieldNormsWriter {
        let mut fieldnorms_buffers: Vec<Option<Vec<u8>>> = iter::repeat_with(|| None)
            .take(schema.num_fields())
            .collect();

        let (static_fields, json_fields) = Self::fields_with_fieldnorm(schema);
        for field in static_fields {
            fieldnorms_buffers[field.field_id() as usize] = Some(Vec::with_capacity(1_000));
        }

        let mut json_fieldnorms_buffers = HashMap::new();
        for field in json_fields {
            json_fieldnorms_buffers.insert(field, HashMap::new());
        }

        FieldNormsWriter {
            fieldnorms_buffers,
            json_fieldnorms_buffers,
        }
    }

    /// The memory used inclusive childs
    pub fn mem_usage(&self) -> usize {
        let regular_usage: usize = self
            .fieldnorms_buffers
            .iter()
            .flatten()
            .map(|buf| buf.capacity())
            .sum();

        let json_usage: usize = self
            .json_fieldnorms_buffers
            .values()
            .map(|buf| buf.values().map(|buf| buf.capacity()).sum::<usize>())
            .sum();

        regular_usage + json_usage
    }

    /// Ensure that all documents in 0..max_doc have a byte associated with them
    /// in each of the fieldnorm vectors.
    ///
    /// Will extend with 0-bytes for documents that have not been seen.
    pub fn fill_up_to_max_doc(&mut self, max_doc: DocId) {
        for fieldnorms_buffer_opt in self.fieldnorms_buffers.iter_mut() {
            if let Some(fieldnorms_buffer) = fieldnorms_buffer_opt.as_mut() {
                fieldnorms_buffer.resize(max_doc as usize, 0u8);
            }
        }

        for field_fieldnorms_buffers in self.json_fieldnorms_buffers.values_mut() {
            field_fieldnorms_buffers
                .values_mut()
                .for_each(|path_fieldnorms_buf| path_fieldnorms_buf.resize(max_doc as usize, 0u8));
        }
    }

    /// Set the fieldnorm byte for the given document for the given field.
    ///
    /// Will internally convert the u32 `fieldnorm` value to the appropriate byte
    /// to approximate the field norm in less space.
    ///
    /// * doc       - the document id
    /// * field     - the field being set
    /// * fieldnorm - the number of terms present in document `doc` in field `field`
    pub fn record(&mut self, doc: DocId, field: Field, fieldnorm: u32) {
        if let Some(fieldnorm_buffer) = self
            .fieldnorms_buffers
            .get_mut(field.field_id() as usize)
            .and_then(Option::as_mut)
        {
            match fieldnorm_buffer.len().cmp(&(doc as usize)) {
                Ordering::Less => {
                    // we fill intermediary `DocId` as  having a fieldnorm of 0.
                    fieldnorm_buffer.resize(doc as usize, 0u8);
                }
                Ordering::Equal => {}
                Ordering::Greater => {
                    panic!("Cannot register a given fieldnorm twice")
                }
            }
            fieldnorm_buffer.push(fieldnorm_to_id(fieldnorm));
        }
    }

    /// Set the fieldnorm byte for the given document for the given JSON field and its path.
    ///
    /// Will internally convert the u32 `fieldnorm` value to the appropriate byte
    /// to approximate the field norm in less space.
    ///
    /// * doc       - the document id
    /// * field     - the field being set
    /// * path      - the JSON path within the field
    /// * fieldnorm - the number of terms present in document `doc` in field `field` at `path`
    pub fn record_json(&mut self, doc: DocId, field: Field, path: &str, fieldnorm: u32) {
        let doc_idx = doc as usize;
        let fieldnorm_buffer = self
            .json_fieldnorms_buffers
            .get_mut(&field)
            .unwrap()
            .raw_entry_mut()
            .from_key(path)
            .or_insert_with(|| {
                (
                    path.to_string(),
                    Vec::with_capacity(std::cmp::max(1000, doc_idx + 1)),
                )
            })
            .1;

        match fieldnorm_buffer.len().cmp(&doc_idx) {
            Ordering::Less => {
                // we fill intermediary `DocId` as having a fieldnorm of 0.
                fieldnorm_buffer.resize(doc_idx, 0u8);
            }
            Ordering::Equal => {}
            Ordering::Greater => {
                // Note: We currently don't handle json arrays properly, each text value in the
                // array will get here with the json path.
                // Current behavior is to skip (first-write wins)
                return;
            }
        }

        fieldnorm_buffer.push(fieldnorm_to_id(fieldnorm));
    }

    /// Serialize the seen fieldnorm values to the serializer for all fields.
    pub fn serialize(&self, mut fieldnorms_serializer: FieldNormsSerializer) -> io::Result<()> {
        for (field, fieldnorms_buffer) in self.fieldnorms_buffers.iter().enumerate().filter_map(
            |(field_id, fieldnorms_buffer_opt)| {
                fieldnorms_buffer_opt.as_ref().map(|fieldnorms_buffer| {
                    (Field::from_field_id(field_id as u32), fieldnorms_buffer)
                })
            },
        ) {
            fieldnorms_serializer.serialize_field(field, fieldnorms_buffer)?;
        }

        for (field, path_map) in &self.json_fieldnorms_buffers {
            for (path, fieldnorms_buffer) in path_map {
                fieldnorms_serializer.serialize_field_path(
                    *field,
                    path.to_string(),
                    fieldnorms_buffer,
                )?;
            }
        }

        fieldnorms_serializer.close()?;
        Ok(())
    }
}
