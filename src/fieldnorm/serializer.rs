use std::io;

use crate::directory::{CompositeWrite, WritePtr};
use crate::fieldnorm::compression::compress_and_write_fieldnorms;
use crate::schema::Field;
use crate::store::Compressor;

/// The fieldnorms serializer is in charge of
/// the serialization of field norms for all fields.
///
/// Uses efficient compression by reusing the compressor instance
/// and batching operations where possible.
pub struct FieldNormsSerializer {
    composite_write: CompositeWrite<WritePtr, String>,
    compressor: Compressor,
    compression_buffer: Vec<u8>,
}

impl FieldNormsSerializer {
    /// Constructor
    pub fn from_write(write: WritePtr, compressor: Compressor) -> io::Result<FieldNormsSerializer> {
        let composite_write: CompositeWrite<WritePtr, String> = CompositeWrite::wrap(write);
        Ok(FieldNormsSerializer {
            composite_write,
            compressor,
            compression_buffer: Vec::new(),
        })
    }

    /// Serialize the given field
    pub fn serialize_field(&mut self, field: Field, fieldnorms_data: &[u8]) -> io::Result<()> {
        let write = self.composite_write.for_field(field);
        compress_and_write_fieldnorms(
            &mut self.compression_buffer,
            &self.compressor,
            write,
            fieldnorms_data,
        )?;

        Ok(())
    }

    /// Serialize the given field and JSON path
    pub fn serialize_field_path(
        &mut self,
        field: Field,
        path: String,
        fieldnorms_data: &[u8],
    ) -> io::Result<()> {
        let write = self.composite_write.for_field_with_idx(field, path);
        compress_and_write_fieldnorms(
            &mut self.compression_buffer,
            &self.compressor,
            write,
            fieldnorms_data,
        )?;

        Ok(())
    }

    /// Clean up / flush / close
    pub fn close(self) -> io::Result<()> {
        self.composite_write.close()?;
        Ok(())
    }
}
