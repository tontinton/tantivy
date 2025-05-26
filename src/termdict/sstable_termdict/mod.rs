use std::io;

mod merger;

use std::iter::ExactSizeIterator;

use common::VInt;
use sstable::value::{ValueReader, ValueWriter};
use sstable::SSTable;
use tantivy_fst::automaton::AlwaysMatch;

pub use self::merger::TermMerger;
use crate::postings::TermInfo;

/// Encoding this as part of the length for backwards compatibility.
/// Old sstables have this bit set to 0, as there cannot be more than u32::MAX term infos).
const IS_RANDOM_ORDER_BIT: u64 = 1 << 31;

/// The term dictionary contains all of the terms in
/// `tantivy index` in a sorted manner.
///
/// The `Fst` crate is used to associate terms to their
/// respective `TermOrdinal`. The `TermInfoStore` then makes it
/// possible to fetch the associated `TermInfo`.
pub type TermDictionary = sstable::Dictionary<TermSSTable>;

/// Builder for the new term dictionary.
pub type TermDictionaryBuilder<W> = sstable::Writer<W, TermInfoValueWriter>;

/// `TermStreamer` acts as a cursor over a range of terms of a segment.
/// Terms are guaranteed to be sorted.
pub type TermStreamer<'a, A = AlwaysMatch> = sstable::Streamer<'a, TermSSTable, A>;

/// SSTable used to store TermInfo objects.
#[derive(Clone)]
pub struct TermSSTable;

pub type TermStreamerBuilder<'a, A = AlwaysMatch> = sstable::StreamerBuilder<'a, TermSSTable, A>;

impl SSTable for TermSSTable {
    type Value = TermInfo;
    type ValueReader = TermInfoValueReader;
    type ValueWriter = TermInfoValueWriter;
}

#[derive(Default)]
pub struct TermInfoValueReader {
    term_infos: Vec<TermInfo>,
}

impl ValueReader for TermInfoValueReader {
    type Value = TermInfo;

    #[inline(always)]
    fn value(&self, idx: usize) -> &TermInfo {
        &self.term_infos[idx]
    }

    fn load(&mut self, mut data: &[u8]) -> io::Result<usize> {
        let len_before = data.len();
        self.term_infos.clear();
        let num_els = VInt::deserialize_u64(&mut data)?;

        let is_random_order = (num_els & IS_RANDOM_ORDER_BIT) != 0;
        let num_els = num_els & !IS_RANDOM_ORDER_BIT;

        if is_random_order {
            for _ in 0..num_els {
                let doc_freq = VInt::deserialize_u64(&mut data)? as u32;
                let postings_start = VInt::deserialize_u64(&mut data)? as usize;
                let postings_num_bytes = VInt::deserialize_u64(&mut data)? as usize;
                let positions_start = VInt::deserialize_u64(&mut data)? as usize;
                let positions_num_bytes = VInt::deserialize_u64(&mut data)? as usize;
                let postings_end = postings_start + postings_num_bytes as usize;
                let positions_end = positions_start + positions_num_bytes as usize;
                let term_info = TermInfo {
                    doc_freq,
                    postings_range: postings_start..postings_end,
                    positions_range: positions_start..positions_end,
                };
                self.term_infos.push(term_info);
            }
            let consumed_len = len_before - data.len();
            return Ok(consumed_len);
        }

        let mut postings_start = VInt::deserialize_u64(&mut data)? as usize;
        let mut positions_start = VInt::deserialize_u64(&mut data)? as usize;
        for _ in 0..num_els {
            let doc_freq = VInt::deserialize_u64(&mut data)? as u32;
            let postings_num_bytes = VInt::deserialize_u64(&mut data)?;
            let positions_num_bytes = VInt::deserialize_u64(&mut data)?;
            let postings_end = postings_start + postings_num_bytes as usize;
            let positions_end = positions_start + positions_num_bytes as usize;
            let term_info = TermInfo {
                doc_freq,
                postings_range: postings_start..postings_end,
                positions_range: positions_start..positions_end,
            };
            self.term_infos.push(term_info);
            postings_start = postings_end;
            positions_start = positions_end;
        }
        let consumed_len = len_before - data.len();
        Ok(consumed_len)
    }
}

#[derive(Default)]
pub struct TermInfoValueWriter {
    term_infos: Vec<TermInfo>,
    encode_random_order: bool,
}

impl ValueWriter for TermInfoValueWriter {
    type Value = TermInfo;

    fn new(encode_random_order: bool) -> Self {
        Self {
            encode_random_order,
            ..Default::default()
        }
    }

    fn write(&mut self, term_info: &TermInfo) {
        self.term_infos.push(term_info.clone());
    }

    fn serialize_block(&self, buffer: &mut Vec<u8>) {
        assert!(self.term_infos.len() < u32::MAX as usize);

        let mut len = self.term_infos.len() as u64;
        if self.encode_random_order {
            len |= IS_RANDOM_ORDER_BIT;
        }

        VInt(len).serialize_into_vec(buffer);
        if self.term_infos.is_empty() {
            return;
        }

        if self.encode_random_order {
            for term_info in &self.term_infos {
                VInt(term_info.doc_freq as u64).serialize_into_vec(buffer);
                VInt(term_info.postings_range.start as u64).serialize_into_vec(buffer);
                VInt(term_info.postings_range.len() as u64).serialize_into_vec(buffer);
                VInt(term_info.positions_range.start as u64).serialize_into_vec(buffer);
                VInt(term_info.positions_range.len() as u64).serialize_into_vec(buffer);
            }
            return;
        }

        VInt(self.term_infos[0].postings_range.start as u64).serialize_into_vec(buffer);
        VInt(self.term_infos[0].positions_range.start as u64).serialize_into_vec(buffer);
        for term_info in &self.term_infos {
            VInt(term_info.doc_freq as u64).serialize_into_vec(buffer);
            VInt(term_info.postings_range.len() as u64).serialize_into_vec(buffer);
            VInt(term_info.positions_range.len() as u64).serialize_into_vec(buffer);
        }
    }

    fn clear(&mut self) {
        self.term_infos.clear();
    }
}

#[cfg(test)]
mod tests {
    use sstable::value::{ValueReader, ValueWriter};

    use crate::postings::TermInfo;
    use crate::termdict::sstable_termdict::TermInfoValueReader;

    #[test]
    fn test_block_terminfos() {
        let mut term_info_writer = super::TermInfoValueWriter::default();
        term_info_writer.write(&TermInfo {
            doc_freq: 120u32,
            postings_range: 17..45,
            positions_range: 10..122,
        });
        term_info_writer.write(&TermInfo {
            doc_freq: 10u32,
            postings_range: 45..450,
            positions_range: 122..1100,
        });
        term_info_writer.write(&TermInfo {
            doc_freq: 17u32,
            postings_range: 450..462,
            positions_range: 1100..1302,
        });
        let mut buffer = Vec::new();
        term_info_writer.serialize_block(&mut buffer);
        let mut term_info_reader = TermInfoValueReader::default();
        let num_bytes: usize = term_info_reader.load(&buffer[..]).unwrap();
        assert_eq!(
            term_info_reader.value(0),
            &TermInfo {
                doc_freq: 120u32,
                postings_range: 17..45,
                positions_range: 10..122
            }
        );
        assert_eq!(buffer.len(), num_bytes);
    }
}
