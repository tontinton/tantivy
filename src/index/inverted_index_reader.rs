use std::io;

use common::json_path_writer::JSON_END_OF_PATH;
use common::{BinarySerializable, OwnedBytes};
use fnv::FnvHashSet;
#[cfg(feature = "quickwit")]
use futures_util::{FutureExt, StreamExt, TryStreamExt};
#[cfg(feature = "quickwit")]
use tantivy_fst::automaton::{AlwaysMatch, Automaton};

use crate::directory::FileSlice;
use crate::positions::PositionReader;
use crate::postings::{BlockSegmentPostings, SegmentPostings, TermInfo};
use crate::schema::{IndexRecordOption, Term, Type};
use crate::termdict::TermDictionary;

// merge holes under 4MiB, that's how many bytes we can hope to receive during a TTFB from
// S3 (~80MiB/s, and 50ms latency)
const MERGE_HOLES_UNDER_BYTES: usize = (80 * 1024 * 1024 * 50) / 1000;

/// The inverted index reader is in charge of accessing
/// the inverted index associated with a specific field.
///
/// # Note
///
/// It is safe to delete the segment associated with
/// an `InvertedIndexReader`. As long as it is open,
/// the [`FileSlice`] it is relying on should
/// stay available.
///
/// `InvertedIndexReader` are created by calling
/// [`SegmentReader::inverted_index()`](crate::SegmentReader::inverted_index).
pub struct InvertedIndexReader {
    termdict: TermDictionary,
    reversed_termdict_opt: Option<TermDictionary>,
    postings_file_slice: FileSlice,
    positions_file_slice: FileSlice,
    record_option: IndexRecordOption,
    total_num_tokens: u64,
}

impl InvertedIndexReader {
    pub(crate) fn new(
        termdict: TermDictionary,
        reversed_termdict_opt: Option<TermDictionary>,
        postings_file_slice: FileSlice,
        positions_file_slice: FileSlice,
        record_option: IndexRecordOption,
    ) -> io::Result<InvertedIndexReader> {
        let (total_num_tokens_slice, postings_body) = postings_file_slice.split(8);
        let total_num_tokens = u64::deserialize(&mut total_num_tokens_slice.read_bytes()?)?;
        Ok(InvertedIndexReader {
            termdict,
            reversed_termdict_opt,
            postings_file_slice: postings_body,
            positions_file_slice,
            record_option,
            total_num_tokens,
        })
    }

    /// Creates an empty `InvertedIndexReader` object, which
    /// contains no terms at all.
    pub fn empty(record_option: IndexRecordOption) -> InvertedIndexReader {
        InvertedIndexReader {
            termdict: TermDictionary::empty(),
            reversed_termdict_opt: None,
            postings_file_slice: FileSlice::empty(),
            positions_file_slice: FileSlice::empty(),
            record_option,
            total_num_tokens: 0u64,
        }
    }

    /// Returns the term info associated with the term.
    pub fn get_term_info(&self, term: &Term) -> io::Result<Option<TermInfo>> {
        self.termdict.get(term.serialized_value_bytes())
    }

    /// Returns the term info associated with the term by looking up the revterm dict.
    pub fn get_term_info_from_revterm(&self, term: &Term) -> io::Result<Option<TermInfo>> {
        let dict = self
            .revterms()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "revterm file doesn't exist"))?;
        Ok(dict.get(term.serialized_value_bytes())?)
    }

    /// Return the term dictionary datastructure.
    pub fn terms(&self) -> &TermDictionary {
        &self.termdict
    }

    /// Return the reversed term dictionary datastructure.
    pub fn revterms(&self) -> Option<&TermDictionary> {
        self.reversed_termdict_opt.as_ref()
    }

    /// Return the reverse term dictionary datastructure if it exists.
    pub fn terms_ext(&self, reverse: bool) -> io::Result<&TermDictionary> {
        if reverse {
            self.revterms()
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "revterm file doesn't exist"))
        } else {
            Ok(self.terms())
        }
    }

    /// Return the fields and types encoded in the dictionary in lexicographic order.
    /// Only valid on JSON fields.
    ///
    /// Notice: This requires a full scan and therefore **very expensive**.
    /// TODO: Move to sstable to use the index.
    pub fn list_encoded_fields(&self) -> io::Result<Vec<(String, Type)>> {
        let mut stream = self.termdict.stream()?;
        let mut fields = Vec::new();
        let mut fields_set = FnvHashSet::default();
        while let Some((term, _term_info)) = stream.next() {
            if let Some(index) = term.iter().position(|&byte| byte == JSON_END_OF_PATH) {
                if !fields_set.contains(&term[..index + 2]) {
                    fields_set.insert(term[..index + 2].to_vec());
                    let typ = Type::from_code(term[index + 1]).unwrap();
                    fields.push((String::from_utf8_lossy(&term[..index]).to_string(), typ));
                }
            }
        }

        Ok(fields)
    }

    /// Resets the block segment to another position of the postings
    /// file.
    ///
    /// This is useful for enumerating through a list of terms,
    /// and consuming the associated posting lists while avoiding
    /// reallocating a [`BlockSegmentPostings`].
    ///
    /// # Warning
    ///
    /// This does not reset the positions list.
    pub fn reset_block_postings_from_terminfo(
        &self,
        term_info: &TermInfo,
        block_postings: &mut BlockSegmentPostings,
    ) -> io::Result<()> {
        let postings_slice = self
            .postings_file_slice
            .slice(term_info.postings_range.clone());
        let postings_bytes = postings_slice.read_bytes()?;
        block_postings.reset(term_info.doc_freq, postings_bytes)?;
        Ok(())
    }

    /// Returns a block postings given a `Term`.
    /// This method is for an advanced usage only.
    ///
    /// Most users should prefer using [`Self::read_postings()`] instead.
    pub fn read_block_postings(
        &self,
        term: &Term,
        option: IndexRecordOption,
    ) -> io::Result<Option<BlockSegmentPostings>> {
        self.get_term_info(term)?
            .map(move |term_info| self.read_block_postings_from_terminfo(&term_info, option))
            .transpose()
    }

    /// Returns a block postings given a `term_info`.
    /// This method is for an advanced usage only.
    ///
    /// Most users should prefer using [`Self::read_postings()`] instead.
    pub fn read_block_postings_from_terminfo(
        &self,
        term_info: &TermInfo,
        requested_option: IndexRecordOption,
    ) -> io::Result<BlockSegmentPostings> {
        let postings_data = self
            .postings_file_slice
            .slice(term_info.postings_range.clone());
        BlockSegmentPostings::open(
            term_info.doc_freq,
            postings_data,
            self.record_option,
            requested_option,
        )
    }

    /// Returns a posting object given a `term_info`.
    /// This method is for an advanced usage only.
    ///
    /// Most users should prefer using [`Self::read_postings()`] instead.
    pub fn read_postings_from_terminfo(
        &self,
        term_info: &TermInfo,
        option: IndexRecordOption,
    ) -> io::Result<SegmentPostings> {
        let option = option.downgrade(self.record_option);

        let block_postings = self.read_block_postings_from_terminfo(term_info, option)?;
        let position_reader = {
            if option.has_positions() {
                let positions_data = self
                    .positions_file_slice
                    .read_bytes_slice(term_info.positions_range.clone())?;
                let position_reader = PositionReader::open(positions_data)?;
                Some(position_reader)
            } else {
                None
            }
        };
        Ok(SegmentPostings::from_block_postings(
            block_postings,
            position_reader,
        ))
    }

    /// Returns the total number of tokens recorded for all documents
    /// (including deleted documents).
    pub fn total_num_tokens(&self) -> u64 {
        self.total_num_tokens
    }

    /// Returns the segment postings associated with the term, and with the given option,
    /// or `None` if the term has never been encountered and indexed.
    ///
    /// If the field was not indexed with the indexing options that cover
    /// the requested options, the returned [`SegmentPostings`] the method does not fail
    /// and returns a `SegmentPostings` with as much information as possible.
    ///
    /// For instance, requesting [`IndexRecordOption::WithFreqs`] for a
    /// [`TextOptions`](crate::schema::TextOptions) that does not index position
    /// will return a [`SegmentPostings`] with `DocId`s and frequencies.
    pub fn read_postings(
        &self,
        term: &Term,
        option: IndexRecordOption,
    ) -> io::Result<Option<SegmentPostings>> {
        self.get_term_info(term)?
            .map(move |term_info| self.read_postings_from_terminfo(&term_info, option))
            .transpose()
    }

    /// Same as read_postings but uses the revterm dict instead of term dict.
    pub fn read_postings_from_revterm(
        &self,
        term: &Term,
        option: IndexRecordOption,
    ) -> io::Result<Option<SegmentPostings>> {
        self.get_term_info_from_revterm(term)?
            .map(move |term_info| self.read_postings_from_terminfo(&term_info, option))
            .transpose()
    }

    /// Returns the number of documents containing the term.
    pub fn doc_freq(&self, term: &Term) -> io::Result<u32> {
        Ok(self
            .get_term_info(term)?
            .map(|term_info| term_info.doc_freq)
            .unwrap_or(0u32))
    }
}

#[cfg(feature = "quickwit")]
impl InvertedIndexReader {
    pub(crate) async fn get_term_info_async(&self, term: &Term) -> io::Result<Option<TermInfo>> {
        self.termdict.get_async(term.serialized_value_bytes()).await
    }

    async fn get_term_range_async<'a, A: Automaton + 'a>(
        &'a self,
        terms: impl std::ops::RangeBounds<Term>,
        automaton: A,
        limit: Option<u64>,
        merge_holes_under_bytes: usize,
        reverse: bool,
    ) -> io::Result<impl Iterator<Item = TermInfo> + 'a>
    where
        A::State: Clone,
    {
        use std::ops::Bound;
        let range_builder = self.terms_ext(reverse)?.search(automaton);
        let range_builder = match terms.start_bound() {
            Bound::Included(bound) => range_builder.ge(bound.serialized_value_bytes()),
            Bound::Excluded(bound) => range_builder.gt(bound.serialized_value_bytes()),
            Bound::Unbounded => range_builder,
        };
        let range_builder = match terms.end_bound() {
            Bound::Included(bound) => range_builder.le(bound.serialized_value_bytes()),
            Bound::Excluded(bound) => range_builder.lt(bound.serialized_value_bytes()),
            Bound::Unbounded => range_builder,
        };
        let range_builder = if let Some(limit) = limit {
            range_builder.limit(limit)
        } else {
            range_builder
        };

        let mut stream = range_builder
            .into_stream_async_merging_holes(merge_holes_under_bytes)
            .await?;

        let iter = std::iter::from_fn(move || stream.next().map(|(_k, v)| v.clone()));

        // limit on stream is only an optimization to load less data, the stream may still return
        // more than limit elements.
        let limit = limit.map(|limit| limit as usize).unwrap_or(usize::MAX);
        let iter = iter.take(limit);

        Ok(iter)
    }

    /// Warmup a block postings given a `Term`.
    /// This method is for an advanced usage only.
    ///
    /// returns a boolean, whether the term was found in the dictionary
    pub async fn warm_postings(&self, term: &Term, with_positions: bool) -> io::Result<bool> {
        let term_info_opt: Option<TermInfo> = self.get_term_info_async(term).await?;
        if let Some(term_info) = term_info_opt {
            let postings = self
                .postings_file_slice
                .read_bytes_slice_async(term_info.postings_range.clone());
            if with_positions {
                let positions = self
                    .positions_file_slice
                    .read_bytes_slice_async(term_info.positions_range.clone());
                futures_util::future::try_join(postings, positions).await?;
            } else {
                postings.await?;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Warmup a block postings given a range of `Term`s.
    /// This method is for an advanced usage only.
    ///
    /// returns a boolean, whether a term matching the range was found in the dictionary
    pub async fn warm_postings_range(
        &self,
        terms: impl std::ops::RangeBounds<Term>,
        limit: Option<u64>,
        with_positions: bool,
    ) -> io::Result<bool> {
        let mut term_info = self
            .get_term_range_async(terms, AlwaysMatch, limit, 0, false)
            .await?;

        let Some(first_terminfo) = term_info.next() else {
            // no key matches, nothing more to load
            return Ok(false);
        };

        let last_terminfo = term_info.last().unwrap_or_else(|| first_terminfo.clone());

        let postings_range = first_terminfo.postings_range.start..last_terminfo.postings_range.end;
        let positions_range =
            first_terminfo.positions_range.start..last_terminfo.positions_range.end;

        let postings = self
            .postings_file_slice
            .read_bytes_slice_async(postings_range);
        if with_positions {
            let positions = self
                .positions_file_slice
                .read_bytes_slice_async(positions_range);
            futures_util::future::try_join(postings, positions).await?;
        } else {
            postings.await?;
        }
        Ok(true)
    }

    #[inline]
    fn close_enough(a: &std::ops::Range<usize>, b: &std::ops::Range<usize>, gap: usize) -> bool {
        a.start <= b.end + gap && b.start <= a.end + gap
    }

    fn coalesce_and_send_ranges<I>(
        stream: I,
        posting_sender: impl Fn(std::ops::Range<usize>) -> std::io::Result<()>,
        positions_sender: impl Fn(std::ops::Range<usize>) -> std::io::Result<()>,
        merge_gap: usize,
    ) -> std::io::Result<()>
    where
        I: Iterator<Item = (std::ops::Range<usize>, std::ops::Range<usize>)>,
    {
        let mut curr_postings: Option<std::ops::Range<usize>> = None;
        let mut curr_positions: Option<std::ops::Range<usize>> = None;

        for (postings_range, positions_range) in stream {
            curr_postings = match curr_postings.take() {
                Some(mut curr) if Self::close_enough(&curr, &postings_range, merge_gap) => {
                    curr.start = curr.start.min(postings_range.start);
                    curr.end = curr.end.max(postings_range.end);
                    Some(curr)
                }
                Some(prev) => {
                    posting_sender(prev)?;
                    Some(postings_range)
                }
                None => Some(postings_range),
            };

            curr_positions = match curr_positions.take() {
                Some(mut curr) if Self::close_enough(&curr, &positions_range, merge_gap) => {
                    curr.start = curr.start.min(positions_range.start);
                    curr.end = curr.end.max(positions_range.end);
                    Some(curr)
                }
                Some(prev) => {
                    positions_sender(prev)?;
                    Some(positions_range)
                }
                None => Some(positions_range),
            };
        }

        if let Some(p) = curr_postings {
            posting_sender(p)?;
        }
        if let Some(pos) = curr_positions {
            positions_sender(pos)?;
        }

        Ok(())
    }

    /// Warmup a block postings given a range of `Term`s.
    /// This method is for an advanced usage only.
    ///
    /// returns a boolean, whether a term matching the range was found in the dictionary
    pub async fn warm_postings_automaton<
        A: Automaton + Clone + Send + 'static,
        E: FnOnce(Box<dyn FnOnce() -> io::Result<()> + Send>) -> F,
        F: std::future::Future<Output = io::Result<()>>,
    >(
        &self,
        automaton: A,
        with_positions: bool,
        reverse: bool,
        executor: E,
    ) -> io::Result<bool>
    where
        A::State: Clone,
    {
        // we build a first iterator to download everything. Simply calling the function already
        // download everything we need from the sstable, but doesn't start iterating over it.
        let _term_info_iter = self
            .get_term_range_async(
                ..,
                automaton.clone(),
                None,
                MERGE_HOLES_UNDER_BYTES,
                reverse,
            )
            .await?;

        let (posting_sender, posting_ranges_to_load_stream) = futures_channel::mpsc::unbounded();
        let (positions_sender, positions_ranges_to_load_stream) =
            futures_channel::mpsc::unbounded();
        let termdict = self.terms_ext(reverse)?.clone();
        let cpu_bound_task = move || {
            // then we build a 2nd iterator, this one with no holes, so we don't go through blocks
            // we can't match.
            // This makes the assumption there is a caching layer below us, which gives sync read
            // for free after the initial async access. This might not always be true, but is in
            // Quickwit.
            // We build things from this closure otherwise we get into lifetime issues that can only
            // be solved with self referential strucs. Returning an io::Result from here is a bit
            // more leaky abstraction-wise, but a lot better than the alternative
            let mut stream = termdict.search(automaton).into_stream()?;

            Self::coalesce_and_send_ranges(
                std::iter::from_fn(move || {
                    stream
                        .next()
                        .map(|(_k, v)| (v.postings_range.clone(), v.positions_range.clone()))
                }),
                |r| {
                    posting_sender
                        .unbounded_send(r)
                        .map_err(|_| std::io::Error::other("failed to send posting range"))
                },
                |r| {
                    if !with_positions {
                        return Ok(());
                    }
                    positions_sender
                        .unbounded_send(r)
                        .map_err(|_| std::io::Error::other("failed to send positions range"))
                },
                MERGE_HOLES_UNDER_BYTES,
            )?;

            Ok(())
        };
        let task_handle = executor(Box::new(cpu_bound_task));

        let posting_downloader = posting_ranges_to_load_stream
            .map(|posting_slice| {
                self.postings_file_slice
                    .read_bytes_slice_async(posting_slice)
                    .map(|result| result.map(|_slice| ()))
            })
            .buffer_unordered(5)
            .try_collect::<Vec<()>>();
        let positions_downloader = positions_ranges_to_load_stream
            .map(|positions_slice| {
                self.positions_file_slice
                    .read_bytes_slice_async(positions_slice)
                    .map(|result| result.map(|_slice| ()))
            })
            .buffer_unordered(5)
            .try_collect::<Vec<()>>();

        let (_, posting_slices_downloaded, positions_slices_downloaded) =
            futures_util::future::try_join3(task_handle, posting_downloader, positions_downloader)
                .await?;

        Ok(!posting_slices_downloaded.is_empty() || !positions_slices_downloaded.is_empty())
    }
    /// Warmup a specific postings file range.
    /// This method is for an advanced usage only.
    pub async fn warm_postings_slice(
        &self,
        range: std::ops::Range<usize>,
    ) -> io::Result<OwnedBytes> {
        self.postings_file_slice.read_bytes_slice_async(range).await
    }

    /// Warmup a specific positions file range.
    /// This method is for an advanced usage only.
    pub async fn warm_positions_slice(
        &self,
        range: std::ops::Range<usize>,
    ) -> io::Result<OwnedBytes> {
        self.positions_file_slice
            .read_bytes_slice_async(range)
            .await
    }

    /// Warmup the block postings for all terms.
    /// This method is for an advanced usage only.
    ///
    /// If you know which terms to pre-load, prefer using [`Self::warm_postings`] or
    /// [`Self::warm_postings`] instead.
    pub async fn warm_postings_full(&self, with_positions: bool) -> io::Result<()> {
        self.postings_file_slice.read_bytes_async().await?;
        if with_positions {
            self.positions_file_slice.read_bytes_async().await?;
        }
        Ok(())
    }

    /// Returns the number of documents containing the term asynchronously.
    pub async fn doc_freq_async(&self, term: &Term) -> io::Result<u32> {
        Ok(self
            .get_term_info_async(term)
            .await?
            .map(|term_info| term_info.doc_freq)
            .unwrap_or(0u32))
    }

    /// Get the termdict file range of a specific term.
    pub fn termdict_file_range_for_term(&self, term: &Term) -> std::ops::Range<usize> {
        self.terms()
            .file_range_for_key(term.serialized_value_bytes())
    }

    /// Get the (postings, positions) file sizes of a block (FST) found by a term.
    pub fn postings_positions_sizes_for_term(&self, term: &Term) -> (u64, Option<(u64, u64)>) {
        self.terms()
            .block_value_sizes_for_key(term.serialized_value_bytes())
    }

    /// Get the file range of the entire postings file.
    pub fn postings_file_range(&self) -> std::ops::Range<usize> {
        self.postings_file_slice.slice_range()
    }

    /// Get the file range of the entire positions file.
    pub fn positions_file_range(&self) -> std::ops::Range<usize> {
        self.positions_file_slice.slice_range()
    }

    /// Get the termdict file range of a term range.
    pub fn termdict_file_range_for_range(
        &self,
        terms: impl std::ops::RangeBounds<Term>,
        limit: Option<u64>,
    ) -> std::ops::Range<usize> {
        let lower = terms.start_bound().map(|b| b.serialized_value_bytes());
        let upper = terms.end_bound().map(|b| b.serialized_value_bytes());

        // warm_postings_range uses AlwaysMatch automaton, meaning we don't need to iterate over
        // specific blocks, and use file_slice_for_range directly.
        // For more info see sstable_delta_reader_for_key_range_async's implementation.
        let slice = self.terms().file_slice_for_range((lower, upper), limit);
        slice.slice_range()
    }

    /// Get the termdict file range of an automaton.
    pub fn termdict_file_ranges_for_automaton<A: Automaton + Clone + Send + 'static>(
        &self,
        automaton: A,
        reverse: bool,
    ) -> io::Result<Vec<std::ops::Range<usize>>>
    where
        A::State: Clone,
    {
        Ok(self
            .terms_ext(reverse)?
            .file_range_for_automaton(&automaton, MERGE_HOLES_UNDER_BYTES)
            .collect())
    }

    /// Get the (postings, positions) file sizes of blocks (FST) found by an automaton.
    pub fn postings_positions_sizes_for_automaton<A: Automaton + Clone + Send + 'static>(
        &self,
        automaton: A,
        reverse: bool,
    ) -> io::Result<Vec<(u64, Option<(u64, u64)>)>>
    where
        A::State: Clone,
    {
        Ok(self
            .terms_ext(reverse)?
            .block_value_sizes_for_automaton(&automaton)
            .collect())
    }
}

#[cfg(all(test, feature = "quickwit"))]
mod test {
    use std::cell::RefCell;

    use super::*;

    #[test]
    fn test_single_merge() {
        let input = vec![(0..10, 100..110), (11..20, 111..120)];
        let merge_gap = 2;

        let postings = RefCell::new(vec![]);
        let positions = RefCell::new(vec![]);

        let posting_sender = |range| {
            postings.borrow_mut().push(range);
            Ok(())
        };
        let positions_sender = |range| {
            positions.borrow_mut().push(range);
            Ok(())
        };

        InvertedIndexReader::coalesce_and_send_ranges(
            input.into_iter(),
            posting_sender,
            positions_sender,
            merge_gap,
        )
        .unwrap();

        assert_eq!(postings.into_inner(), vec![0..20]);
        assert_eq!(positions.into_inner(), vec![100..120]);
    }

    #[test]
    fn test_no_merge() {
        let input = vec![
            (0..10, 100..110),
            (20..30, 120..130), // gap too large
        ];
        let merge_gap = 5;

        let postings = RefCell::new(vec![]);
        let positions = RefCell::new(vec![]);

        let posting_sender = |range| {
            postings.borrow_mut().push(range);
            Ok(())
        };
        let positions_sender = |range| {
            positions.borrow_mut().push(range);
            Ok(())
        };

        InvertedIndexReader::coalesce_and_send_ranges(
            input.into_iter(),
            posting_sender,
            positions_sender,
            merge_gap,
        )
        .unwrap();

        assert_eq!(postings.into_inner(), vec![0..10, 20..30]);
        assert_eq!(positions.into_inner(), vec![100..110, 120..130]);
    }

    #[test]
    fn test_multiple_merges_and_flush_at_end() {
        let input = vec![(0..10, 100..110), (11..15, 111..115), (30..40, 130..140)];
        let merge_gap = 2;

        let postings = RefCell::new(vec![]);
        let positions = RefCell::new(vec![]);

        let posting_sender = |range| {
            postings.borrow_mut().push(range);
            Ok(())
        };
        let positions_sender = |range| {
            positions.borrow_mut().push(range);
            Ok(())
        };

        InvertedIndexReader::coalesce_and_send_ranges(
            input.into_iter(),
            posting_sender,
            positions_sender,
            merge_gap,
        )
        .unwrap();

        assert_eq!(postings.into_inner(), vec![0..15, 30..40]);
        assert_eq!(positions.into_inner(), vec![100..115, 130..140]);
    }

    #[test]
    fn test_empty_input() {
        let input: Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> = vec![];
        let merge_gap = 1;

        let postings = RefCell::new(vec![]);
        let positions = RefCell::new(vec![]);

        let posting_sender = |range| {
            postings.borrow_mut().push(range);
            Ok(())
        };
        let positions_sender = |range| {
            positions.borrow_mut().push(range);
            Ok(())
        };

        InvertedIndexReader::coalesce_and_send_ranges(
            input.into_iter(),
            posting_sender,
            positions_sender,
            merge_gap,
        )
        .unwrap();

        assert!(postings.borrow().is_empty());
        assert!(positions.borrow().is_empty());
    }

    #[test]
    fn test_unsorted_merge() {
        let input = vec![(11..20, 111..120), (0..10, 100..110)]; // reverse order
        let merge_gap = 2;

        let postings = RefCell::new(vec![]);
        let positions = RefCell::new(vec![]);

        let posting_sender = |r| {
            postings.borrow_mut().push(r);
            Ok(())
        };
        let positions_sender = |r| {
            positions.borrow_mut().push(r);
            Ok(())
        };

        InvertedIndexReader::coalesce_and_send_ranges(
            input.into_iter(),
            posting_sender,
            positions_sender,
            merge_gap,
        )
        .unwrap();

        assert_eq!(postings.into_inner(), vec![0..20]);
        assert_eq!(positions.into_inner(), vec![100..120]);
    }

    #[test]
    fn test_unsorted_no_merge() {
        let input = vec![(20..30, 120..130), (0..10, 100..110)]; // unsorted
        let merge_gap = 5; // gap < 10 --> no merge

        let postings = RefCell::new(vec![]);
        let positions = RefCell::new(vec![]);

        let posting_sender = |r| {
            postings.borrow_mut().push(r);
            Ok(())
        };
        let positions_sender = |r| {
            positions.borrow_mut().push(r);
            Ok(())
        };

        InvertedIndexReader::coalesce_and_send_ranges(
            input.into_iter(),
            posting_sender,
            positions_sender,
            merge_gap,
        )
        .unwrap();

        assert_eq!(postings.into_inner(), vec![20..30, 0..10]);
        assert_eq!(positions.into_inner(), vec![120..130, 100..110]);
    }

    #[test]
    fn test_unsorted_overlap_merge() {
        let input = vec![(20..25, 120..125), (10..30, 110..140)];
        let merge_gap = 0; // direct overlap

        let postings = RefCell::new(vec![]);
        let positions = RefCell::new(vec![]);

        let posting_sender = |r| {
            postings.borrow_mut().push(r);
            Ok(())
        };
        let positions_sender = |r| {
            positions.borrow_mut().push(r);
            Ok(())
        };

        InvertedIndexReader::coalesce_and_send_ranges(
            input.into_iter(),
            posting_sender,
            positions_sender,
            merge_gap,
        )
        .unwrap();

        assert_eq!(postings.into_inner(), vec![10..30]);
        assert_eq!(positions.into_inner(), vec![110..140]);
    }
}
