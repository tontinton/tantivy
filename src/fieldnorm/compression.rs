use std::io;
use std::io::Write;

use common::{BinarySerializable, CountingWriter};

use crate::directory::WritePtr;
use crate::store::{Compressor, Decompressor};

#[derive(Debug, Clone)]
pub struct FieldNormsHeader {
    decompressor_id: u8,
    original_size: u32,
    compressed_size: u32,
}

/// Serializes the header to a byte-array
/// - decompressor id: 1 byte
/// - original size: 4 bytes
/// - compressed size: 4 bytes
impl BinarySerializable for FieldNormsHeader {
    fn serialize<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        BinarySerializable::serialize(&self.decompressor_id, writer)?;
        BinarySerializable::serialize(&self.original_size, writer)?;
        BinarySerializable::serialize(&self.compressed_size, writer)?;
        Ok(())
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        Ok(FieldNormsHeader {
            decompressor_id: BinarySerializable::deserialize(reader)?,
            original_size: BinarySerializable::deserialize(reader)?,
            compressed_size: BinarySerializable::deserialize(reader)?,
        })
    }
}

pub fn compress_and_write_fieldnorms(
    compression_buffer: &mut Vec<u8>,
    compressor: &Compressor,
    write: &mut CountingWriter<WritePtr>,
    fieldnorms_data: &[u8],
) -> io::Result<()> {
    compression_buffer.clear();
    compressor.compress_into(fieldnorms_data, compression_buffer)?;
    let header = FieldNormsHeader {
        decompressor_id: crate::store::Decompressor::from(*compressor).get_id(),
        original_size: fieldnorms_data.len() as u32,
        compressed_size: compression_buffer.len() as u32,
    };
    BinarySerializable::serialize(&header, write)?;
    write.write_all(compression_buffer)?;
    write.flush()?;

    Ok(())
}

pub fn decompress_fieldnorms(data: &[u8]) -> crate::Result<Vec<u8>> {
    let mut reader = std::io::Cursor::new(data);
    let header = FieldNormsHeader::deserialize(&mut reader)?;
    let data_start = reader.position() as usize;
    let compressed_data = &data[data_start..data_start + header.compressed_size as usize];
    let decompressor = Decompressor::from_id(header.decompressor_id);
    Ok(decompressor.decompress(compressed_data)?)
}
