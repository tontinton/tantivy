use std::io;

mod merger;

use std::iter::ExactSizeIterator;

use common::{VInt, MERGE_HOLES_UNDER_BYTES};
use itertools::Itertools;
use sstable::value::{BlockValueSizes, ValueReader, ValueWriter};
use sstable::SSTable;
use tantivy_fst::automaton::AlwaysMatch;

pub use self::merger::TermMerger;
use crate::postings::TermInfo;

/// Encoding this as part of the length for backwards compatibility.
/// Old sstables have this bit set to 0, as there cannot be more than u32::MAX term infos).
const IS_RANDOM_ORDER_BIT: u64 = 1 << 31;

/// VInt encodes a number by simply dividing by 128, until reaching a number smaller than 128,
/// and encoding the final byte with the highest bit turned on.
/// When encoding a set of large numbers, we can instead encode the smallest of the set, and then
/// use VInt only on the diff, which effectively means we store (N / 128, where N is our smallest
/// number in the set) less bytes per item.
const START_ADDRESS_OPTIMIZATION_BIT: u64 = 1 << 32;

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

#[inline]
fn extract_flag(num: u64, bit: u64) -> (u64, bool) {
    (num & !bit, (num & bit) != 0)
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

        let (num_els, is_random_order) = extract_flag(num_els, IS_RANDOM_ORDER_BIT);
        let (num_els, is_start_address_optimization) =
            extract_flag(num_els, START_ADDRESS_OPTIMIZATION_BIT);

        if is_random_order {
            let (
                min_doc_freq,
                min_postings_start,
                min_postings_num_bytes,
                min_positions_start,
                min_positions_num_bytes,
            ) = if is_start_address_optimization {
                (
                    VInt::deserialize_u64(&mut data)? as u32,
                    VInt::deserialize_u64(&mut data)? as usize,
                    VInt::deserialize_u64(&mut data)? as usize,
                    VInt::deserialize_u64(&mut data)? as usize,
                    VInt::deserialize_u64(&mut data)? as usize,
                )
            } else {
                (0, 0, 0, 0, 0)
            };

            for _ in 0..num_els {
                let doc_freq = VInt::deserialize_u64(&mut data)? as u32 + min_doc_freq;
                let postings_start =
                    VInt::deserialize_u64(&mut data)? as usize + min_postings_start;
                let postings_num_bytes =
                    VInt::deserialize_u64(&mut data)? as usize + min_postings_num_bytes;
                let positions_start =
                    VInt::deserialize_u64(&mut data)? as usize + min_positions_start;
                let positions_num_bytes =
                    VInt::deserialize_u64(&mut data)? as usize + min_positions_num_bytes;
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

        let (min_doc_freq, min_postings_num_bytes, min_positions_num_bytes) =
            if is_start_address_optimization {
                (
                    VInt::deserialize_u64(&mut data)? as u32,
                    VInt::deserialize_u64(&mut data)?,
                    VInt::deserialize_u64(&mut data)?,
                )
            } else {
                (0, 0, 0)
            };

        let mut postings_start = VInt::deserialize_u64(&mut data)? as usize;
        let mut positions_start = VInt::deserialize_u64(&mut data)? as usize;
        for _ in 0..num_els {
            let doc_freq = VInt::deserialize_u64(&mut data)? as u32 + min_doc_freq;
            let postings_num_bytes = VInt::deserialize_u64(&mut data)? + min_postings_num_bytes;
            let positions_num_bytes = VInt::deserialize_u64(&mut data)? + min_positions_num_bytes;
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
    dont_optimize_start_address: bool,
    block_sizes: BlockValueSizes,
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
        self.block_sizes.postings_size += term_info.posting_num_bytes() as u64;
        self.block_sizes.positions_size += term_info.positions_num_bytes() as u64;
    }

    fn serialize_block(&self, buffer: &mut Vec<u8>) {
        assert!(self.term_infos.len() < u32::MAX as usize);

        let mut len = self.term_infos.len() as u64;
        if self.encode_random_order {
            len |= IS_RANDOM_ORDER_BIT;
        }
        if !self.dont_optimize_start_address {
            len |= START_ADDRESS_OPTIMIZATION_BIT;
        }

        VInt(len).serialize_into_vec(buffer);
        if self.term_infos.is_empty() {
            return;
        }

        if self.encode_random_order {
            let (
                min_doc_freq,
                min_postings_start,
                min_postings_len,
                min_positions_start,
                min_positions_len,
            ) = if !self.dont_optimize_start_address {
                let mut min_doc_freq = u64::MAX;
                let mut min_postings_start = u64::MAX;
                let mut min_postings_len = u64::MAX;
                let mut min_positions_start = u64::MAX;
                let mut min_positions_len = u64::MAX;

                for term_info in &self.term_infos {
                    min_doc_freq = min_doc_freq.min(term_info.doc_freq as u64);
                    min_postings_start =
                        min_postings_start.min(term_info.postings_range.start as u64);
                    min_postings_len = min_postings_len.min(term_info.postings_range.len() as u64);
                    min_positions_start =
                        min_positions_start.min(term_info.positions_range.start as u64);
                    min_positions_len =
                        min_positions_len.min(term_info.positions_range.len() as u64);
                }

                VInt(min_doc_freq as u64).serialize_into_vec(buffer);
                VInt(min_postings_start as u64).serialize_into_vec(buffer);
                VInt(min_postings_len as u64).serialize_into_vec(buffer);
                VInt(min_positions_start as u64).serialize_into_vec(buffer);
                VInt(min_positions_len as u64).serialize_into_vec(buffer);

                (
                    min_doc_freq,
                    min_postings_start,
                    min_postings_len,
                    min_positions_start,
                    min_positions_len,
                )
            } else {
                (0, 0, 0, 0, 0)
            };

            for term_info in &self.term_infos {
                VInt(term_info.doc_freq as u64 - min_doc_freq).serialize_into_vec(buffer);
                VInt(term_info.postings_range.start as u64 - min_postings_start)
                    .serialize_into_vec(buffer);
                VInt(term_info.postings_range.len() as u64 - min_postings_len)
                    .serialize_into_vec(buffer);
                VInt(term_info.positions_range.start as u64 - min_positions_start)
                    .serialize_into_vec(buffer);
                VInt(term_info.positions_range.len() as u64 - min_positions_len)
                    .serialize_into_vec(buffer);
            }
            return;
        }

        let (min_doc_freq, min_postings_len, min_positions_len) = if !self
            .dont_optimize_start_address
        {
            let mut min_doc_freq = u64::MAX;
            let mut min_postings_len = u64::MAX;
            let mut min_positions_len = u64::MAX;

            for term_info in &self.term_infos {
                min_doc_freq = min_doc_freq.min(term_info.doc_freq as u64);
                min_postings_len = min_postings_len.min(term_info.postings_range.len() as u64);
                min_positions_len = min_positions_len.min(term_info.positions_range.len() as u64);
            }

            VInt(min_doc_freq as u64).serialize_into_vec(buffer);
            VInt(min_postings_len as u64).serialize_into_vec(buffer);
            VInt(min_positions_len as u64).serialize_into_vec(buffer);

            (min_doc_freq, min_postings_len, min_positions_len)
        } else {
            (0, 0, 0)
        };

        VInt(self.term_infos[0].postings_range.start as u64).serialize_into_vec(buffer);
        VInt(self.term_infos[0].positions_range.start as u64).serialize_into_vec(buffer);
        for term_info in &self.term_infos {
            VInt(term_info.doc_freq as u64 - min_doc_freq).serialize_into_vec(buffer);
            VInt(term_info.postings_range.len() as u64 - min_postings_len)
                .serialize_into_vec(buffer);
            VInt(term_info.positions_range.len() as u64 - min_positions_len)
                .serialize_into_vec(buffer);
        }
    }

    fn block_value_sizes(&self) -> Option<BlockValueSizes> {
        let (postings_ranges, positions_ranges): (Vec<_>, Vec<_>) = self
            .term_infos
            .iter()
            .map(|t| (t.postings_range.clone(), t.positions_range.clone()))
            .unzip();

        let (postings_size, postings_range_start, postings_range_end) =
            merged_ranges_size_and_boundaries(postings_ranges);
        let (positions_size, positions_range_start, positions_range_end) =
            merged_ranges_size_and_boundaries(positions_ranges);

        let mut block_sizes = self.block_sizes.clone();
        block_sizes.coalesced_postings_size = postings_size as u64;
        block_sizes.coalesced_positions_size = positions_size as u64;
        block_sizes.postings_range_start = postings_range_start;
        block_sizes.postings_range_end = postings_range_end;
        block_sizes.positions_range_start = positions_range_start;
        block_sizes.positions_range_end = positions_range_end;
        Some(block_sizes)
    }

    fn clear(&mut self) {
        self.term_infos.clear();
        self.block_sizes = BlockValueSizes::default();
    }
}

fn merged_ranges_size_and_boundaries(mut ranges: Vec<std::ops::Range<usize>>) -> (usize, u64, u64) {
    if ranges.is_empty() {
        return (0, 0, 0);
    }

    ranges.sort_by_key(|r| r.start);

    let range_start = ranges[0].start as u64;
    let range_end = ranges.last().unwrap().end as u64;

    let merged_size = ranges
        .into_iter()
        .coalesce(|first, second| {
            if first.end + MERGE_HOLES_UNDER_BYTES >= second.start {
                Ok(first.start..second.end)
            } else {
                Err((first, second))
            }
        })
        .map(|range| range.len())
        .sum();

    (merged_size, range_start, range_end)
}

#[cfg(test)]
mod tests {
    use sstable::value::{ValueReader, ValueWriter};

    use crate::postings::TermInfo;
    use crate::termdict::sstable_termdict::TermInfoValueReader;

    fn check_block_terminfos(mut term_info_writer: super::TermInfoValueWriter) {
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

    #[test]
    fn test_block_terminfos() {
        check_block_terminfos(super::TermInfoValueWriter::default());
    }

    #[test]
    fn test_block_terminfos_backwards_compatibility() {
        check_block_terminfos(super::TermInfoValueWriter {
            dont_optimize_start_address: true,
            ..Default::default()
        });
    }

    #[test]
    fn test_block_terminfos_random_order() {
        let mut term_info_writer = super::TermInfoValueWriter::new(true);
        term_info_writer.write(&TermInfo {
            doc_freq: 5u32,
            postings_range: 100..150,
            positions_range: 200..260,
        });
        term_info_writer.write(&TermInfo {
            doc_freq: 10u32,
            postings_range: 300..400,
            positions_range: 450..500,
        });
        term_info_writer.write(&TermInfo {
            doc_freq: 20u32,
            postings_range: 500..530,
            positions_range: 600..640,
        });

        let mut buffer = Vec::new();
        term_info_writer.serialize_block(&mut buffer);

        let mut term_info_reader = super::TermInfoValueReader::default();
        let consumed = term_info_reader.load(&buffer).unwrap();

        assert_eq!(consumed, buffer.len());

        let term_infos = &term_info_reader.term_infos;
        assert_eq!(term_infos.len(), 3);
        assert_eq!(
            term_infos[0],
            TermInfo {
                doc_freq: 5,
                postings_range: 100..150,
                positions_range: 200..260,
            }
        );
        assert_eq!(
            term_infos[1],
            TermInfo {
                doc_freq: 10,
                postings_range: 300..400,
                positions_range: 450..500,
            }
        );
        assert_eq!(
            term_infos[2],
            TermInfo {
                doc_freq: 20,
                postings_range: 500..530,
                positions_range: 600..640,
            }
        );
    }

    #[test]
    fn test_block_value_sizes_with_ranges() {
        let mut term_info_writer = super::TermInfoValueWriter::default();

        term_info_writer.write(&TermInfo {
            doc_freq: 10u32,
            postings_range: 200..250,
            positions_range: 350..450,
        });

        term_info_writer.write(&TermInfo {
            doc_freq: 20u32,
            postings_range: 100..150,
            positions_range: 200..250,
        });

        term_info_writer.write(&TermInfo {
            doc_freq: 15u32,
            postings_range: 350..400,
            positions_range: 500..600,
        });

        let sizes = term_info_writer.block_value_sizes().unwrap();

        assert_eq!(sizes.postings_range_start, 100);
        assert_eq!(sizes.postings_range_end, 400);
        assert_eq!(sizes.positions_range_start, 200);
        assert_eq!(sizes.positions_range_end, 600);
    }

    #[test]
    fn test_block_value_sizes_empty_term_infos() {
        let term_info_writer = super::TermInfoValueWriter::default();
        let sizes = term_info_writer.block_value_sizes().unwrap();

        assert_eq!(sizes.postings_range_start, 0);
        assert_eq!(sizes.postings_range_end, 0);
        assert_eq!(sizes.positions_range_start, 0);
        assert_eq!(sizes.positions_range_end, 0);
    }
}
