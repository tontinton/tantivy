pub(crate) mod index;
mod range;
mod u64_monotonic;
mod void;

use std::io;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct BlockValueSizes {
    pub postings_size: u64,
    pub positions_size: u64,
    pub coalesced_postings_size: u64,
    pub coalesced_positions_size: u64,
}

/// `ValueReader` is a trait describing the contract of something
/// reading blocks of value, and offering random access within this values.
pub trait ValueReader: Default {
    /// Type of the value being read.
    type Value;

    /// Access the value at index `idx`, in the last block that was read
    /// via a call to `ValueReader::read`.
    fn value(&self, idx: usize) -> &Self::Value;

    /// Loads a block.
    ///
    /// Returns the number of bytes that were read.
    fn load(&mut self, data: &[u8]) -> io::Result<usize>;
}

/// `ValueWriter` is a trait to make it possible to write blocks
/// of value.
pub trait ValueWriter: Default {
    /// Type of the value being written.
    type Value;

    /// Values are written sequentially ordered, if the value writer uses this information to more
    /// compactly encode the values, but you need to store non sequential values, call this
    /// function with _encode_random_order=true.
    fn new(_encode_random_order: bool) -> Self {
        Self::default()
    }

    /// Records a new value.
    /// This method usually just accumulates data in a `Vec`,
    /// only to be serialized on the call to `ValueWriter::serialize_block`.
    fn write(&mut self, val: &Self::Value);

    /// Serializes the accumulated values into the output buffer.
    fn serialize_block(&self, output: &mut Vec<u8>);

    /// Get a block's postings and positions size.
    fn block_value_sizes(&self) -> Option<BlockValueSizes> {
        None
    }

    /// Clears the `ValueWriter`. After a call to clear, the `ValueWriter`
    /// should behave like a fresh `ValueWriter::default()`.
    fn clear(&mut self);
}

pub use range::{RangeValueReader, RangeValueWriter};
pub use u64_monotonic::{U64MonotonicValueReader, U64MonotonicValueWriter};
pub use void::{VoidValueReader, VoidValueWriter};

fn deserialize_vint_u64(data: &mut &[u8]) -> u64 {
    let (num_bytes, val) = super::vint::deserialize_read(data);
    *data = &data[num_bytes..];
    val
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fmt;

    use super::{ValueReader, ValueWriter};

    pub(crate) fn test_value_reader_writer<
        V: Eq + fmt::Debug,
        TReader: ValueReader<Value = V>,
        TWriter: ValueWriter<Value = V>,
    >(
        value_block: &[V],
    ) {
        let mut buffer = Vec::new();
        {
            let mut writer = TWriter::default();
            for value in value_block {
                writer.write(value);
            }
            writer.serialize_block(&mut buffer);
            writer.clear();
        }
        let data_len = buffer.len();
        buffer.extend_from_slice(&b"extradata"[..]);
        let mut reader = TReader::default();
        assert_eq!(reader.load(&buffer[..]).unwrap(), data_len);
        for (i, val) in value_block.iter().enumerate() {
            assert_eq!(reader.value(i), val);
        }
    }
}
