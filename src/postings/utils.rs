use common::BitSet;

use super::TermInfo;
use crate::schema::IndexRecordOption;
use crate::{DocSet, InvertedIndexReader, TERMINATED};

pub fn collect_docs_for_term_info(
    inverted_index: &InvertedIndexReader,
    term_info: &TermInfo,
    doc_bitset: &mut BitSet,
    must_start: bool,
) -> crate::Result<()> {
    if must_start {
        collect_docs_starting_with_term_info(inverted_index, term_info, doc_bitset)
    } else {
        collect_all_docs_for_term_info(inverted_index, term_info, doc_bitset)
    }
}

/// Collects documents where the term appears at position 0
fn collect_docs_starting_with_term_info(
    inverted_index: &InvertedIndexReader,
    term_info: &TermInfo,
    doc_bitset: &mut BitSet,
) -> crate::Result<()> {
    let mut postings = inverted_index
        .read_postings_from_terminfo(term_info, IndexRecordOption::WithFreqsAndPositions)?;

    let mut doc = postings.doc();
    while doc != TERMINATED {
        if postings.has_position_zero() {
            doc_bitset.insert(doc);
        }
        doc = postings.advance();
    }

    Ok(())
}

/// Collects all documents with term
fn collect_all_docs_for_term_info(
    inverted_index: &InvertedIndexReader,
    term_info: &TermInfo,
    doc_bitset: &mut BitSet,
) -> crate::Result<()> {
    let mut block_segment_postings =
        inverted_index.read_block_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;

    loop {
        let docs = block_segment_postings.docs();
        if docs.is_empty() {
            break;
        }

        for &doc in docs {
            doc_bitset.insert(doc);
        }

        block_segment_postings.advance();
    }

    Ok(())
}
