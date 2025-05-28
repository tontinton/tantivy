use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use common::json_path_writer::JSON_END_OF_PATH;
use itertools::kmerge_by;
use tempfile::TempDir;

use crate::postings::{FieldSerializer, TermInfo};
use crate::schema::Type;

/// Write to memory until reaching this size, then flush to disk.
/// 100MB.
const FLUSH_SIZE: usize = 100 * 1024 * 1024;

/// Size of the BufReader for each file read to be merged.
/// Because there could be a lot of files (merged_termdict_size / FLUSH_SIZE), keep this low while
/// not too low, so merging can go quicker.
/// 4KB.
const BUFREADER_SIZE: usize = 4096;

pub struct RevTermMerger {
    dir: TempDir,
    terms: Vec<(Vec<u8>, TermInfo)>,
    tracked_size: usize,
    files_flushed: usize,
}

impl RevTermMerger {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            dir: TempDir::new()?,
            terms: Vec::new(),
            tracked_size: 0,
            files_flushed: 0,
        })
    }

    /// Currently only supports json string fields.
    pub fn push(&mut self, bytes: &[u8], info: TermInfo) -> io::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }

        let Some(pos) = bytes.iter().cloned().position(|b| b == JSON_END_OF_PATH) else {
            return Ok(());
        };
        let (prefix, value) = bytes.split_at(pos + 1);

        if value.is_empty() || value[0] != Type::Str as u8 {
            return Ok(());
        }

        let mut rev = Vec::with_capacity(bytes.len());
        rev.extend(prefix);
        rev.push(value[0]);
        rev.extend(value.iter().skip(1).rev().copied());
        self.terms.push((rev, info));

        self.tracked_size += bytes.len() + size_of::<TermInfo>();
        if self.tracked_size >= FLUSH_SIZE {
            self.flush()?;
        }

        Ok(())
    }

    #[inline]
    fn get_path(&self, id: usize) -> PathBuf {
        self.dir.path().join(format!("{}.rev", id))
    }

    #[inline]
    fn sort_terms(&mut self) {
        self.terms.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.terms.is_empty() {
            return Ok(());
        }

        self.sort_terms();

        let file = File::create(self.get_path(self.files_flushed))?;
        let mut writer = BufWriter::new(file);
        for (term, info) in &self.terms {
            writer.write_all(&(term.len() as u32).to_le_bytes())?;
            writer.write_all(term)?;
            writer.write_all(&info.doc_freq.to_le_bytes())?;
            writer.write_all(&(info.postings_range.start as u64).to_le_bytes())?;
            writer.write_all(&(info.postings_range.len() as u32).to_le_bytes())?;
            writer.write_all(&(info.positions_range.start as u64).to_le_bytes())?;
            writer.write_all(&(info.positions_range.len() as u32).to_le_bytes())?;
        }
        writer.flush()?;

        self.terms.clear();
        self.tracked_size = 0;
        self.files_flushed += 1;

        Ok(())
    }

    pub fn merge(mut self, serializer: &mut FieldSerializer<'_>) -> io::Result<()> {
        if self.files_flushed == 0 {
            // No file written, just use what's in memory.

            self.sort_terms();
            for (term, info) in &self.terms {
                serializer.insert_reversed_term(term, info)?;
            }
            return Ok(());
        }

        self.flush()?;
        self.terms.shrink_to_fit();

        let mut revterm_iters = Vec::with_capacity(self.files_flushed);

        for i in 0..self.files_flushed {
            let file = File::open(self.get_path(i))?;
            let reader = BufReader::with_capacity(BUFREADER_SIZE, file);
            revterm_iters.push(RevTermIterator::new(reader));
        }

        let merge_iter = kmerge_by(
            revterm_iters,
            |a: &io::Result<(Vec<u8>, TermInfo)>, b: &io::Result<(Vec<u8>, TermInfo)>| {
                match (a, b) {
                    (Ok((a_term, _)), Ok((b_term, _))) => a_term.cmp(b_term) == Ordering::Less,

                    // Propagate the error.
                    (Err(_), _) => true,
                    (_, Err(_)) => false,
                }
            },
        );

        for result in merge_iter {
            let (term, info) = result?;
            serializer.insert_reversed_term(&term, &info)?;
        }

        Ok(())
    }
}

struct RevTermIterator<R: Read> {
    reader: R,
}

impl<R: Read> RevTermIterator<R> {
    fn new(reader: R) -> Self {
        RevTermIterator { reader }
    }
}

impl<R: Read> Iterator for RevTermIterator<R> {
    type Item = Result<(Vec<u8>, TermInfo), io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let term_len = read_u32(&mut self.reader).ok()? as usize;

        let mut term = vec![0u8; term_len as usize];
        if let Err(e) = self.reader.read_exact(&mut term) {
            return Some(Err(unexpected_eof_reading("term", e)));
        }

        Some(
            read_term_info(&mut self.reader)
                .map(|term_info| (term, term_info))
                .map_err(|e| unexpected_eof_reading("TermInfo", e)),
        )
    }
}

fn unexpected_eof_reading(msg: &str, e: io::Error) -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        format!("Unexpected EOF while reading {}: {}", msg, e),
    )
}

fn read_term_info<R: Read>(reader: &mut R) -> io::Result<TermInfo> {
    let doc_freq = read_u32(reader)?;
    let postings_start = read_u64(reader)? as usize;
    let postings_len = read_u32(reader)? as usize;
    let positions_start = read_u64(reader)? as usize;
    let positions_len = read_u32(reader)? as usize;

    Ok(TermInfo {
        doc_freq,
        postings_range: postings_start..postings_start + postings_len,
        positions_range: positions_start..positions_start + positions_len,
    })
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
